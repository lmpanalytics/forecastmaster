/*
 * Copyright 2017 LMP Consulting.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package se.bigdatamining.forecastmaster.query;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import javax.inject.Named;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.PostConstruct;
import javax.enterprise.context.RequestScoped;
import javax.inject.Inject;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.neo4j.driver.v1.Record;
import org.neo4j.driver.v1.Session;
import org.neo4j.driver.v1.StatementResult;
import org.neo4j.driver.v1.Values;
import org.neo4j.driver.v1.exceptions.ClientException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.bigdatamining.forecastmaster.Neo4jBean;
import se.bigdatamining.forecastmaster.train.Train;

/**
 * Queries a previously trained RNN for time series predictions
 *
 * @author Magnus Palm
 */
@Named(value = "queryBean")
@RequestScoped // Import latest data from @PostConstruct on call 'doQuery' from jsf or on refresh
public class QueryBean implements Serializable {

    @Inject
    Neo4jBean neo4jBean;

    @Inject
    Train trainer;

    private static final long serialVersionUID = 1L;
    private static final Logger LOGGER = LoggerFactory.getLogger(QueryBean.class);
    private static Session session;
    private static INDArray newPredicted;

    //Initiate fields for training, testing, and time steps. Set in @PostConstruct.
    private static int previousStateTrainSize = 0;
//    private static int testSize = 0;
    private int numberOfTimesteps = 0;
    private static int miniBatchSize = 0;
    private double[] forecasts;
    private int additionalFuturePredictions;

    /**
     * Creates a new instance of QueryBean
     */
    public QueryBean() {
    }

    @PostConstruct
    public void init() {
        // INITIATE CLASS SPECIFIC MAPS AND FIELDS HERE - THE ORDER IS IMPORTANT

        // Initialize driver
        this.session = neo4jBean.getDRIVER().session();

        // Get trainer fields
        previousStateTrainSize = trainer.getTrainSize();
//        testSize = trainer.getTestSize();
        numberOfTimesteps = trainer.getNumberOfTimesteps();
        miniBatchSize = trainer.getMiniBatchSize();

        additionalFuturePredictions = getQualifiedAdditionalFuturePredictions(); // These are predictions t+2, t+3, ..., t+n
        forecasts = new double[1 + additionalFuturePredictions];
    }

    private static File initBaseFile(String fileName) {
        try {
            return new ClassPathResource(fileName).getFile();
        } catch (IOException e) {
            throw new Error(e);
        }
    }

    private static File baseDir = initBaseFile("/rnnRegression");
    private static File baseTrainDir = new File(baseDir, "multiTimestepTrainQuery");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "multiTimestepTestQuery");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    private static int numOfVariables = 0;  // in csv.

    /**
     * Queries the trained RNN for future time predictions.
     *
     * @throws Exception
     */
    public void doQuery() throws Exception {

        LOGGER.info("Using AbsolutePath: " + baseDir.getAbsolutePath());

        //Set number of examples for training, testing, and time steps
//        int trainSize = 43 - 20; // (Time slices) Same as length of CSV file, i.e. rawStrings.size - numberOfTimesteps
        int trainSize = queryRawDataSizeDB() - numberOfTimesteps; // (Time slices) Same as length of CSV file, i.e. rawStrings.size - numberOfTimesteps
//        int numberOfTimesteps = 20; // Must be same as for the trained RMM

        //Prepare multi time step data, see method comments for more info
        List<String> rawStrings = getQueryData(trainSize, numberOfTimesteps);

        //Make sure miniBatchSize is divisable by trainSize and testSize,
        //as rnnTimeStep will not accept different sized examples
//        int miniBatchSize = 10;
        // ----- Load the training data -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize - 1));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize - 1));

        DataSetIterator trainDataIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        LOGGER.info("*****LOAD FITTED NORMALIZER*****");
        // Where to load fitted normalizer from
        File locationToSaveNormalizer = new File("fitted_normalizer.zip");

        // Restore the normalizer from the stored file.
        NormalizerSerializer serializer = NormalizerSerializer.getDefault();
        NormalizerMinMaxScaler normalizer = serializer.restore(locationToSaveNormalizer);

        normalizer.fitLabel(true);

        trainDataIter.reset();

        trainDataIter.setPreProcessor(normalizer);

        LOGGER.info("*****LOAD TRAINED MODEL*****");
        // Where to load model from
        File locationToSave = new File("trained_rnn_model.zip");

        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        // Import and load the state from the trained rnn
        net.rnnSetPreviousState(0, getStateFile());

        // Get the predictions
        DataSet t = trainDataIter.next();
        INDArray parentFeatureArray = t.getFeatureMatrix(); // up to and including idx 128
        INDArray predicted = net.rnnTimeStep(parentFeatureArray);
//        System.out.println("parentFeatureArray: " + parentFeatureArray.toString());
//        System.out.println("");
        /* normalizer.revertLabels(predicted);
        System.out.println("predicted: " + predicted.toString());
        System.out.println(""); */

        // ****** Prepare to predict using data after the initial data processing is done *****
        // Convert the list of rawstrings to an array of doubles
        double[][] conversionArray = new double[rawStrings.size()][numOfVariables]; //create an array with the size of the rawStrings

        for (int i = 0; i < rawStrings.size(); ++i) { //iterate over the elements of the list
            for (int j = 0; j < numOfVariables; j++) {
                String[] featureArray = rawStrings.get(i).split(",");
                conversionArray[i][j] = Double.parseDouble(featureArray[j]); //store each element as a double in the array
            }
        }

        // Convert to INDArray
        INDArray featureArray = Nd4j.create(conversionArray);

        // Normalize
        normalizer.transform(featureArray);

        // ***** Prepare to loop through the remaining data in the rawstring list *****
        // Get the shape
        int[] shape = parentFeatureArray.shape();

        // get a range of rows (row a (inclusive) to row b (exclusive)) and all columns
        // effectively same as deleting the top row
        INDArray subsetArray = parentFeatureArray.get(NDArrayIndex.interval(1, miniBatchSize), NDArrayIndex.all()).dup();

        // copy the bottom row as an independent row
        INDArray bottomRow = parentFeatureArray.getRow(miniBatchSize - 1).dup();

        // Loop through the remaining time slices after last feature in QueryRNN.java (memory)
        int loopStartIdx = 28 + 1; // Previous state + 1 idx
        int delta = rawStrings.size() - loopStartIdx; // Entries until end of list
        for (int i = loopStartIdx; i < loopStartIdx + delta; i++) {

            // left shift the bottom row vector one step
            shiftRowOneStep(numberOfTimesteps, bottomRow);

            // pick row containing next pair of features to process
            INDArray row = featureArray.getRows(i);

            // get last prediction and use as features to process
            double[] scalar = new double[numOfVariables];
            for (int j = 0; j < numOfVariables; j++) {
                scalar[j] = row.getDouble(j);
                // add new values at the end of the bottom row
                bottomRow.putScalar(j, numberOfTimesteps - 1, scalar[j]);
            }

            // flatten subsetArray and flatten the bottomRow, and concatenate them together
            INDArray flattened = Nd4j.toFlattened(subsetArray, bottomRow);

            // recast the flattened array to a new feature array
            INDArray newFeatureArray = flattened.reshape('c', shape);
//            System.out.println("newFeatureArray: " + newFeatureArray.toString());

            // feed to 'net.rnnTimeStep()' and get updated predictions for next time slice
            newPredicted = net.rnnTimeStep(newFeatureArray);

            // get a range of rows (row a (inclusive) to row b (exclusive)) and all columns
            // effectively same as deleting the top row
            subsetArray = newFeatureArray.get(NDArrayIndex.interval(1, miniBatchSize), NDArrayIndex.all()).dup();

            // copy the bottom row as an independent row
            bottomRow = newFeatureArray.getRow(miniBatchSize - 1).dup();

            // View prediction t+1 in the last feature loop
            if (i == loopStartIdx + delta - 1 && additionalFuturePredictions == 0) {
                normalizer.revertLabels(newPredicted);
//                System.out.println("newPredicted: " + newPredicted.toString());

                // put prediction t+1 in forecast output array
                bottomRow = newPredicted.getRow(miniBatchSize - 1).dup();
                INDArray pred = bottomRow.getScalar(0, numberOfTimesteps - 1);
                forecasts[0] = pred.getDouble(0);

                LOGGER.info("SUCCESS: Generated forecast for one period");
            }
        }

        // ***** Feed the future prediction back in the loop *****
        for (int i = 0; i < additionalFuturePredictions; i++) {

            INDArray newPredictedBottomRow = newPredicted.getRow(miniBatchSize - 1).dup();

            // left shift the bottom row vector one step
            shiftRowOneStep(numberOfTimesteps, bottomRow);

            // get last prediction and use as features to process
            double[] predArray = new double[numOfVariables];
            for (int j = 0; j < numOfVariables; j++) {
                INDArray pred = newPredictedBottomRow.getScalar(j, numberOfTimesteps - 1);
                predArray[j] = pred.getDouble(0);
                // add new values at the end of the bottom row
                bottomRow.putScalar(j, numberOfTimesteps - 1, predArray[j]);
            }

// flatten subsetArray and flatten the bottomRow, and concatenate them together
            INDArray flattened = Nd4j.toFlattened(subsetArray, bottomRow);

            // recast the flattened array to a new feature array
            INDArray newFeatureArray = flattened.reshape('c', shape);
//            System.out.println("newFeatureArray: " + newFeatureArray.toString());

            // feed to 'net.rnnTimeStep()' and get updated predictions for next time slice
            newPredicted = net.rnnTimeStep(newFeatureArray);

            // get a range of rows (row a (inclusive) to row b (exclusive)) and all columns
            // effectively same as deleting the top row
            subsetArray = newFeatureArray.get(NDArrayIndex.interval(1, miniBatchSize), NDArrayIndex.all()).dup();

            // copy the bottom row as an independent row
            bottomRow = newFeatureArray.getRow(miniBatchSize - 1).dup();

            // View prediction t+n in the last feature loop
            if (i == additionalFuturePredictions - 1) {
                normalizer.revertLabels(newPredicted);
//                System.out.println("newPredicted: " + newPredicted.toString());

                // put predictions in forecast output array
                newPredictedBottomRow = newPredicted.getRow(miniBatchSize - 1).dup();
                for (int j = 0; j < additionalFuturePredictions + 1; j++) {
                    INDArray pred = newPredictedBottomRow.getScalar(0, numberOfTimesteps - 1 - additionalFuturePredictions + j);
                    forecasts[j] = pred.getDouble(0);
                }
                LOGGER.info("SUCCESS: Generated forecast for {} periods", additionalFuturePredictions + 1);
            }
        }
    }

    /**
     * left shift the bottom row vector one step
     */
    private static void shiftRowOneStep(int numberOfTimesteps, INDArray bottomRow) {
        // left shift the bottom row vector one step
        for (int m = 0; m < numOfVariables; m++) {
            for (int n = 0; n < numberOfTimesteps; n++) {
                double value = 0;
                if (n < numberOfTimesteps - 1) {
                    value = bottomRow.getDouble(m, n + 1);
                }
                bottomRow.putScalar(m, n, value);
            }
        }
    }

    /**
     * Method to import the RNN time step's previous state from file on disk
     */
    private static Map<String, INDArray> getStateFile() {
        Map<String, INDArray> rnnPreviousState = new HashMap<>();
        try {
            try (FileInputStream fi = new FileInputStream(new File("rnnPreviousState.txt"));
                    ObjectInputStream oi = new ObjectInputStream(fi)) {
                // Read objects from file
                rnnPreviousState = (Map<String, INDArray>) oi.readObject();
            }
            LOGGER.info("SUCCESS: RNN Previous state read from file");

        } catch (FileNotFoundException e) {
            LOGGER.error("File not found. {}", e);
        } catch (IOException e) {
            LOGGER.error("Error initializing stream. {}", e);
        } catch (ClassNotFoundException e) {
            LOGGER.error("Class not found. {}", e);
        }
        return rnnPreviousState;
    }

    /**
     * This method, modeled on the 'prepareTrainAndTest()' method, reads raw
     * data from a CSV file containing:
     *
     * 1/ Time series data leading up to the previous stored RNN time step
     * state. 2/ Future time series data to be processed. The CSV file starts
     * from the test data in the CSV file used for training and testing the RNN.
     */
    private static List<String> getQueryData(int trainSize, int numberOfTimesteps) throws IOException {
        Path rawPath = Paths.get(baseDir.getAbsolutePath() + "/data_0000000001_prev_state_raw.csv");

        //Remove data files before generating new one from database
        // List all files in baseDir folder
        File folder = new File(baseDir.getAbsolutePath());
        File fList[] = folder.listFiles();
        // Searches ... prev_state_raw.csv
        for (int i = 0; i < fList.length; i++) {
            File f = fList[i];
            if (f.getName().endsWith("prev_state_raw.csv")) {
                // and deletes
                File target = fList[i];
                boolean success = target.delete();
                if (success) {
                    LOGGER.info("Deleted data file {} ", f.toPath().toAbsolutePath().toString());
                } else {
                    LOGGER.error("Delete data file {} ", f.toPath().toAbsolutePath().toString());
                }

            }
        }

        //Remove all files before generating new ones
        FileUtils.cleanDirectory(featuresDirTrain);
        FileUtils.cleanDirectory(labelsDirTrain); // FIX LOGIC
        FileUtils.cleanDirectory(featuresDirTest);
        FileUtils.cleanDirectory(labelsDirTest);

        generateDataFile(rawPath);

        List<String> rawStrings = Files.readAllLines(rawPath, Charset.defaultCharset());
        setNumOfVariables(rawStrings);

        for (int i = 0; i < trainSize; i++) {
            Path featuresPath = Paths.get(featuresDirTrain.getAbsolutePath() + "/train_" + i + ".csv");
            Path labelsPath = Paths.get(labelsDirTrain + "/train_" + i + ".csv");
            for (int j = 0; j < numberOfTimesteps; j++) {
                Files.write(featuresPath, rawStrings.get(i + j).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);

//                String var = rawStrings.get(i + j);
//                System.out.printf("Time sliceIdx %s, SampleIdx %s : %s%n", i, j, var);
            }
            // FIX NEEDED. CREATE DUMMY LABELS INSTEAD?
            Files.write(labelsPath, rawStrings.get(i + numberOfTimesteps).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }
        return rawStrings;
    }

    /**
     * Queries database for customer's raw data. Skips the training set and
     * returns data that is post 'Previous State'. Writes a csv data file unique
     * to the customer incorporating the customer number in the file name, like
     * so 'data_0000000001_prev_state_raw.csv'.
     *
     * @param rawPath to the file location
     */
    private static void generateDataFile(Path rawPath) {
        try {

            // Extract the customer number from the raw path
            String fileName = rawPath.getFileName().toString();
            int idx = fileName.indexOf("_");
            String customerNumber = fileName.substring(idx + 1, idx + 11);

            String tx = "MATCH (p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}) RETURN p.msEpoch AS ms, p.respVar0 AS respVar0 ORDER BY ms SKIP $skip";

            StatementResult result = session.run(tx, Values.parameters(
                    "custNo", customerNumber,
                    "skip", previousStateTrainSize
            ));
            while (result.hasNext()) {
                Record next = result.next();

                String respVar0 = Double.toString(next.get("respVar0").asDouble());

                // Add results to data file unique to the customer
                Files.write(rawPath, respVar0.concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }
            LOGGER.info("SUCCESS: Wrote CSV data file {}", rawPath);
        } catch (ClientException e) {
            LOGGER.error("ClientException in 'generateCSV' {}", e);
        } catch (IOException ex) {
            LOGGER.error("IOException in 'generateCSV' {}", ex);
        }
    }

    /**
     * Query the Database for the number of patterns in the rawdata for a
     * specific customer. Skips the training set and returns data that is post
     * 'Previous State'. The size is used as input for the method getQueryData,
     * which writes the CSV raw data file.
     *
     * @return size of rawdata
     */
    private int queryRawDataSizeDB() {
        int count = 0;
        try {

            String customerNumber = "0000000001";

            String tx = "MATCH (p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}) RETURN COUNT(p) AS dataSize";

            StatementResult result = session.run(tx, Values.parameters(
                    "custNo", customerNumber
            ));
            while (result.hasNext()) {
                Record next = result.next();

                count = next.get("dataSize").asInt() - previousStateTrainSize;

            }
        } catch (Exception e) {
            LOGGER.error("Could not query the Raw Data Size from the DB");
        }

        return count;
    }

    private static void setNumOfVariables(List<String> rawStrings) {
        numOfVariables = rawStrings.get(0).split(",").length;
    }

    /**
     * Method to import the qualifiedAdditionalFuturePredictions from file on
     * disk.
     *
     * These are predictions t+2, t+3, ..., t+n
     *
     * (Prediction t+1 is always given)
     */
    private static int getQualifiedAdditionalFuturePredictions() {
        int addFcPeriods = 0;
        try {
            try (FileInputStream fi = new FileInputStream(new File("qualifiedAdditionalFuturePredictions.txt"));
                    ObjectInputStream oi = new ObjectInputStream(fi)) {
                // Read objects from file
                addFcPeriods = (int) oi.readObject();
            }
            LOGGER.info("SUCCESS: Qualified additional future predictions read from file");

        } catch (FileNotFoundException e) {
            LOGGER.error("File not found. {}", e);
        } catch (IOException e) {
            LOGGER.error("Error initializing stream. {}", e);
        } catch (ClassNotFoundException e) {
            LOGGER.error("Class not found. {}", e);
        }
        return addFcPeriods;
    }

    /**
     * Get the forecast produced by the RNN. If the last actual value is at time
     * t0, then the forecast array will contain predictions for t+1,
     * t+2,...,t+n. The number of time periods to forecast after t+1 is set by
     * class field 'additionalFuturePredictions'.
     *
     * @return array of forecast values
     */
    public double[] getForecasts() {
        return forecasts;
    }
}

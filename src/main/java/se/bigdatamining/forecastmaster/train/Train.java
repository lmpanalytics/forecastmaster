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
package se.bigdatamining.forecastmaster.train;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import javax.annotation.PostConstruct;
import javax.enterprise.context.SessionScoped;
import javax.inject.Inject;
import javax.inject.Named;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.neo4j.driver.v1.Record;
import org.neo4j.driver.v1.Session;
import org.neo4j.driver.v1.StatementResult;
import org.neo4j.driver.v1.Transaction;
import org.neo4j.driver.v1.Values;
import static org.neo4j.driver.v1.Values.parameters;
import org.neo4j.driver.v1.exceptions.ClientException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.bigdatamining.forecastmaster.Neo4jBean;

/**
 * The class runs a regression on multiple time steps. The data is transposed
 * from a single csv file containing one our multiple columns (variables). The
 * first column is the data to be predicted.
 *
 * The number of columns are detected automatically. The csv structure is the
 * master, i.e., if you want to regress on more or less columns - add or remove
 * columns in the csv file, do not change class variables.
 *
 * @author Magnus Palm
 */
@Named(value = "train")
@SessionScoped
public class Train implements Serializable {

    @Inject
    Neo4jBean neo4jBean;

    @Inject
    ProgressBarView progressBarView;

    private static final long serialVersionUID = 1L;
    private static final Logger LOGGER = LoggerFactory.getLogger(Train.class);
    private static Session session;

    //Set number of examples for training, testing, and time steps
    private static int trainSize = 100;
    private static int testSize = 20;
    private static int numberOfTimesteps = 20;
    //Make sure miniBatchSize is divisable by trainSize and testSize,
    //as rnnTimeStep will not accept different sized examples
    private static int miniBatchSize = 10;

    @PostConstruct
    public void init() {
        // INITIATE CLASS SPECIFIC MAPS AND FIELDS HERE - THE ORDER IS IMPORTANT

        // Initialize driver
        this.session = neo4jBean.getDRIVER().session();

    }

    // (for development purpose) Roll 'Predicted test data' series forward in plot
    private static final int SLIDE = 0;

    private static File initBaseFile(String fileName) {
        try {
            return new ClassPathResource(fileName).getFile();
        } catch (IOException e) {
            throw new Error(e);
        }
    }

    private static File baseDir = initBaseFile("/rnnRegression");
    private static File baseTrainDir = new File(baseDir, "multiTimestepTrain");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "multiTimestepTest");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    private static int numOfVariables = 0;  // in csv.

    /**
     * Prepares raw data, configures, trains and evaluates the RNN, serializes
     * and saves the Network Configuration, Weights matrix, Normalizer
     * statistics and Previous state, collects predictions generated by test
     * data, sets progress of the Training progress bar.
     *
     * @throws Exception
     */
    public void doTraining() throws Exception {

        List<Double> predictions = new LinkedList<>();

        LOGGER.info("Using AbsolutePath: " + baseDir.getAbsolutePath());

        //Prepare multi time step data, see method comments for more info
        List<String> rawStrings = prepareTrainAndTest(trainSize, testSize, numberOfTimesteps);

        // ----- Load the training data -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize - 1));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize - 1));

        DataSetIterator trainDataIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Normalize the training data
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainDataIter);              //Collect training data statistics
        trainDataIter.reset();

        LOGGER.info("*****SAVE FITTED NORMALIZER*****");

        // Where to save normalizer
        File locationToSaveNormalizer = new File("fitted_normalizer.zip");

        // Now we want to save the normalizer to a binary file. For doing this, one can use the NormalizerSerializer.
        NormalizerSerializer serializer = NormalizerSerializer.getDefault();

        // Save the normalizer to the location to save
        serializer.write(normalizer, locationToSaveNormalizer);

        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/test_%d.csv", trainSize, trainSize + testSize - 1));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/test_%d.csv", trainSize, trainSize + testSize - 1));

        DataSetIterator testDataIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        trainDataIter.setPreProcessor(normalizer);
        testDataIter.setPreProcessor(normalizer);

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(140)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.15)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numOfVariables).nOut(10)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).nIn(10).nOut(numOfVariables).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));

        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 50;

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainDataIter);
            trainDataIter.reset();
            LOGGER.info("Epoch " + i + " complete. Time series evaluation:");

            RegressionEvaluation evaluation = new RegressionEvaluation(numOfVariables);

            //Run evaluation. This is on 25k reviews, so can take some time
            while (testDataIter.hasNext()) {
                DataSet t = testDataIter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray labels = t.getLabels();
                INDArray predicted = net.output(features, true);

                evaluation.evalTimeSeries(labels, predicted);
            }

            System.out.println(evaluation.stats());
            testDataIter.reset();
        }

        LOGGER.info("*****SAVE TRAINED MODEL*****");
        // Details

        // Where to save model
        File locationToSave = new File("trained_rnn_model.zip");

        // boolean save Updater
        boolean saveUpdater = false;

        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

        /*
         * All code below this point is only necessary for plotting
         */
        //Initialize rnnTimeStep with train data
        while (trainDataIter.hasNext()) {
            DataSet t = trainDataIter.next();
            net.rnnTimeStep(t.getFeatureMatrix());
        }

// Get the rnnTimeStep state after Initialized rnnTimeStep with train data and
// export to file / DB
        Map<String, INDArray> state = net.rnnGetPreviousState(0);
        exportStateFile(state);

        trainDataIter.reset();

        // Predict test data using the rnnTimeStep method
        DataSet t = testDataIter.next();
        INDArray predicted = net.rnnTimeStep(t.getFeatureMatrix());
        normalizer.revertLabels(predicted);
        // Identical predictions as from 'INDArray predicted = net.output(features, true)'
//        System.out.println("predicted: " + predicted.toString());

// Collect predictions to list for DB writing
        INDArray arrayOfPredictions = predicted.getRow(0);
        for (int i = 0; i < arrayOfPredictions.length(); i++) {
            predictions.add(arrayOfPredictions.getDouble(i));
        }

        // Write predictions to DB
        writePredictionsToDB(predictions);

        /*
        //Convert raw string data to IndArrays for plotting
        INDArray trainArray = createIndArrayFromStringList(rawStrings, 0, trainSize);
        INDArray testArray = createIndArrayFromStringList(rawStrings, trainSize, testSize);

        //Create plot with out data
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainArray, 0, "Train data");
        createSeries(c, testArray, trainSize, "Actual test data");
        createSeries(c, predicted, trainSize + 1 + SLIDE, "Predicted test data");

        plotDataset(c); */
        // Fast-forward the Training Progress Bar to 100% when training method is ready
        progressBarView.setProgress(100);
        LOGGER.info("----- Example Complete -----");
    }

    /**
     * Creates an IndArray from a list of strings Used for plotting purposes
     */
    private static INDArray createIndArrayFromStringList(List<String> rawStrings, int startIndex, int length) {
        List<String> stringList = rawStrings.subList(startIndex, startIndex + length);

        double[][] primitives = new double[numOfVariables][stringList.size()];
        for (int i = 0; i < stringList.size(); i++) {
            String[] vals = stringList.get(i).split(",");
            for (int j = 0; j < vals.length; j++) {
                primitives[j][i] = Double.valueOf(vals[j]);
            }
        }

        return Nd4j.create(new int[]{1, length}, primitives);
    }

    /**
     * Used to create the different time series for plotting purposes
     */
    private static void createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = data.shape()[2];
        boolean predicted = name.startsWith("Predicted");

        XYSeries series = new XYSeries(name);
        // temp hack to accomodate single column case
        if (numOfVariables == 1) {
            for (int i = 0; i < nRows; i++) {
                series.add(i + offset, data.getDouble(i));
            }
        } else /* multiple column case */ {
            for (int i = 0; i < nRows; i++) {
                if (predicted) {
                    series.add(i + offset, data.slice(SLIDE).slice(0).getDouble(i));
                } else {
                    series.add(i + offset, data.slice(0).getDouble(i));
                }
            }
        }
        seriesCollection.addSeries(series);
    }

    /**
     * Generate an xy plot of the datasets provided.
     */
    private static void plotDataset(XYSeriesCollection c) {

        String title = "Regression example";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Number of passengers";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
    }

    /**
     * This method shows how you based on a CSV file can preprocess your data
     * the structure expected for a multi time step problem. This examples uses
     * a single column CSV as input, but the example should be easy to modify
     * for use with a multi column input as well.
     *
     * @return list of Strings like input rows in the CSV file, e.g.,
     * "112,95.91917291".
     * @throws IOException
     */
    private static List<String> prepareTrainAndTest(int trainSize, int testSize, int numberOfTimesteps) throws IOException {

        Path rawPath = Paths.get(baseDir.getAbsolutePath() + "/data_0000000001_raw.csv");

        //Remove data files before generating new one from database
        // List all files in baseDir folder
        File folder = new File(baseDir.getAbsolutePath());
        File fList[] = folder.listFiles();
        // Searches .csv
        for (int i = 0; i < fList.length; i++) {
            File f = fList[i];
            if (f.getName().endsWith("raw.csv")) {
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
        FileUtils.cleanDirectory(labelsDirTrain);
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
            }
            Files.write(labelsPath, rawStrings.get(i + numberOfTimesteps).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }

        for (int i = trainSize; i < testSize + trainSize; i++) {
            Path featuresPath = Paths.get(featuresDirTest + "/test_" + i + ".csv");
            Path labelsPath = Paths.get(labelsDirTest + "/test_" + i + ".csv");
            for (int j = 0; j < numberOfTimesteps; j++) {
                Files.write(featuresPath, rawStrings.get(i + j).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }
            Files.write(labelsPath, rawStrings.get(i + numberOfTimesteps).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }

        return rawStrings;
    }

    private static void setNumOfVariables(List<String> rawStrings) {
        numOfVariables = rawStrings.get(0).split(",").length;
    }

    /**
     * Method to export the RNN time step's previous state to file on disk
     *
     * @param rnnPreviousState the state to export
     */
    private static void exportStateFile(Map<String, INDArray> rnnPreviousState) {
        try {
            FileOutputStream f = new FileOutputStream(new File("rnnPreviousState.txt"));
            // Write objects to file
            try (ObjectOutputStream o = new ObjectOutputStream(f)) {
                // Write objects to file
                o.writeObject(rnnPreviousState);
                LOGGER.info("SUCCESS: RNN Previous state written to file 'rnnPreviousState.txt'");
            }
        } catch (FileNotFoundException e) {
            LOGGER.error("File not found. {}", e);
        } catch (IOException e) {
            LOGGER.error("Error initializing stream. {}", e);
        }
    }

    /**
     * Queries database for customer's raw data. Writes a csv data file unique
     * to the customer incorporating the customer number in the file name, like
     * so 'data_0000000001_raw.csv'.
     *
     * @param rawPath to the file location
     */
    private static void generateDataFile(Path rawPath) {
        try {

            // Extract the customer number from the raw path
            String fileName = rawPath.getFileName().toString();
            int idx = fileName.indexOf("_");
            String customerNumber = fileName.substring(idx + 1, idx + 11);

            String tx = "MATCH (p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}) RETURN p.msEpoch AS ms, p.respVar0 AS respVar0 ORDER BY ms";

            StatementResult result = session.run(tx, Values.parameters(
                    "custNo", customerNumber
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
     * Deletes any existing predictions before adding new ones for the test set.
     *
     * @param predictions
     */
    private void writePredictionsToDB(List<Double> predictions) {

        String customerNumber = "0000000001";

        // Delete existing predictions
        String s1 = "MATCH (p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}), (p)<-[r]-(pr) DELETE r, pr";

        try (Transaction tx = session.beginTransaction()) {
            tx.run(s1, parameters("custNo", customerNumber));
            tx.success();  // Mark this write as successful.
        }

        // Create new predictions
        String s2 = "MATCH (p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}) WITH p ORDER BY p.msEpoch SKIP $skip LIMIT 1"
                + " MERGE (p)<-[:PREDICTED_FOR]-(pr:Prediction {predicted:$pred})";

        // Loop through the predictions
        if (!predictions.isEmpty()) {
            for (int i = 0; i < predictions.size(); i++) {
                Double predValue = predictions.get(i);

                try (Transaction tx = session.beginTransaction()) {

                    tx.run(s2, parameters(
                            "custNo", customerNumber,
                            "skip", trainSize + 1 + i,
                            "pred", predValue
                    ));

                    tx.success();  // Mark this write as successful.
                }
            }
            LOGGER.info("SUCCESS: Wrote {} predictions for customer {} to DB.", predictions.size(), customerNumber);
        } else {
            LOGGER.error("Prediction collection is empty.");
        }
    }
}
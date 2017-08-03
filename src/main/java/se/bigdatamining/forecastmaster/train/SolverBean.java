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

import javax.annotation.PostConstruct;
import javax.inject.Named;
import javax.enterprise.context.RequestScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Qualifies various splits in sizes of raw data for training a neural net.
 *
 * In general, the smaller the rawDataSize the smaller the miniBatchSize to find
 * a divisable solution. However, too small data sets have a negative impact on
 * RNN performance, therefore, recommend bigger data sets (n > 30).
 *
 * NOTE: Smallest rawDataSize for the Solver class to converge to a solution is:
 * 5.
 *
 * @author Magnus Palm
 */
@Named(value = "solverBean")
@RequestScoped
public class SolverBean {

    private static final Logger LOGGER = LoggerFactory.getLogger(SolverBean.class);

    // Constants
    private static final double TRAINING_SHARE_UL = 80 / 100d /*Upper limit*/;
    private static final double TEST_SHARE_UL = 20 / 100d /*Upper limit*/;

    // Fields initiated by Setters in Training class
    private Long rawDataSize = 0L;

    private Long trainSize = 0L;
    private Long testSize = 0L;
    private Long numberOfTimesteps = 0L /* Looking back, cannot be larger than the training size */;

    // Fields initiated by @PostConstruct
    private static Long miniBatchSize /* Must be divisable by trainSize and testSize  */;
    private static double trainingShareLL /*Lower limit*/;
    private static double testShareLL /*Lower limit*/;
    private static boolean isFoundSolution;

    /**
     * Creates a new instance of SolverBean
     */
    public SolverBean() {
    }

    @PostConstruct
    public void init() {
        // INITIATE CLASS SPECIFIC MAPS AND FIELDS HERE - THE ORDER IS IMPORTANT

        isFoundSolution = false;

// ***** Initialize trainingShareLL and testShareLL *****
        if (rawDataSize >= 150L) {
            trainingShareLL = 70 / 100d;
            testShareLL = 15 / 100d;
        } else {
            trainingShareLL = 65 / 100d;
            testShareLL = 10 / 100d;
        }

        // ***** Initialize miniBatchSize size *****
        if (rawDataSize > 1L && rawDataSize < 15L) {
            miniBatchSize = 1L;
        } else if (rawDataSize >= 15L && rawDataSize < 40L) {
            miniBatchSize = 2L;
        } else if (rawDataSize >= 40L && rawDataSize < 80L) {
            miniBatchSize = 5L;
        } else {
            // Continue to build the ladder...
            miniBatchSize = 10L;
        }
    }

    /**
     * Find a solver solution
     */
    public boolean solve() {

        System.out.println("\nSOLVER WORKING...\n------------------------------------------------------------------------");

// ***** 1. Calculate training size *****
        boolean isTrainSizeDivisable = solveTrainingSize();

        // ***** 2. Calculate test size *****
        boolean isTestSizeDivisable = solveTestSize();

        // ***** 3. Check so all sum up *****
        Long validationSize = rawDataSize - trainSize - testSize;
        Long sumTotal = trainSize + testSize + validationSize;

        boolean isDiffZero = false;
        if (sumTotal - rawDataSize == 0) {
            isDiffZero = true;
        }

        // ***** 4. Check so Number of Timesteps Fit in Train Size *****
        boolean isNumberOfTimestepsFit = checkTimeStepFit();

        System.out.printf("%n***** SOLVER RESULTS *****%nRaw Data Size:\t%s%nTraining size:\t%s%nTest size:\t%s%nValidation size:\t%s%n"
                + "SumTotal - RawDataSize:\t%s%nNumber of Timesteps:\t%s%nminiBatchSize:\t%s%nDiff to RawData Size is Zero:\t%s%n"
                + "Number of Timesteps Fit in Train Size:\t%s%nTraining Size is Divisable by MiniBatch:\t%s%n"
                + "Test Size is Divisable by MiniBatch:\t%s%n------------------------------------------------------------------------%n",
                rawDataSize, trainSize, testSize, validationSize, sumTotal - rawDataSize, numberOfTimesteps, miniBatchSize, isDiffZero, isNumberOfTimestepsFit, isTrainSizeDivisable, isTestSizeDivisable);

        // If all booleans are true, return true, else false
        if (isDiffZero && isNumberOfTimestepsFit && isTrainSizeDivisable && isTestSizeDivisable) {
            isFoundSolution = true;
            LOGGER.info("SUCCESS: Solver found solution");
        } else {
            LOGGER.error("Solver failed to find solution");
        }

        return isFoundSolution;
    }

    private boolean checkTimeStepFit() {

        boolean isNumberOfTimestepsFit = false;
        if (numberOfTimesteps <= trainSize) {
            isNumberOfTimestepsFit = true;
        } else {
            System.out.printf("numberOfTimesteps:\t%s%n", numberOfTimesteps);
            while (!isNumberOfTimestepsFit && numberOfTimesteps > 0) {
                numberOfTimesteps--;

                if (numberOfTimesteps <= trainSize) {
                    isNumberOfTimestepsFit = true;
                    System.out.println("Adjusted numberOfTimesteps: " + numberOfTimesteps);
                }
            }
            if (!isNumberOfTimestepsFit) {
                System.err.println("ERROR: No suitable numberOfTimesteps found");
            }
        }
        return isNumberOfTimestepsFit;
    }

    private boolean solveTestSize() {

        testSize = Math.round(testShareLL * rawDataSize);
        // Optimize Test Size
        // Divisable by miniBatchSize
        boolean isTestSizeDivisable = false;
        if (testSize % miniBatchSize == 0 && testSize > 0) {
            isTestSizeDivisable = true;
        } else {
            System.out.printf("TestSizeDivisableRest:\t%s%n", testSize % miniBatchSize);
            while (testSize % miniBatchSize != 0 && testSize <= TEST_SHARE_UL * rawDataSize && trainSize + testSize < rawDataSize && testSize > 0) {
                testSize++;

                if (testSize % miniBatchSize == 0 && testSize <= TEST_SHARE_UL * rawDataSize && trainSize + testSize < rawDataSize && testSize > 0) {
                    isTestSizeDivisable = true;
                    System.out.println("Adjusted testSize: " + testSize);
                }
            }
            if (!isTestSizeDivisable) {
                System.err.println("ERROR: No suitable test size found");
            }

        }
        return isTestSizeDivisable;
    }

    private boolean solveTrainingSize() {

        trainSize = Math.round(trainingShareLL * rawDataSize);
        // Optimize Train Size
        // Divisable by miniBatchSize
        boolean isTrainSizeDivisable = false;
        if (trainSize % miniBatchSize == 0 && trainSize > 0) {
            isTrainSizeDivisable = true;
        } else {
            System.out.printf("TrainSizeDivisableRest:\t%s%n", trainSize % miniBatchSize);
            while (trainSize % miniBatchSize != 0 && trainSize <= TRAINING_SHARE_UL * rawDataSize && trainSize > 0) {
                trainSize++;

                if (trainSize % miniBatchSize == 0 && trainSize <= TRAINING_SHARE_UL * rawDataSize && trainSize > 0) {
                    isTrainSizeDivisable = true;
                    System.out.println("Adjusted trainSize: " + trainSize);
                }
            }
            if (!isTrainSizeDivisable) {
                System.err.println("ERROR: No suitable train size found");
            }

        }
        return isTrainSizeDivisable;
    }

    public void setRawDataSize(Long rawDataSize) {
        this.rawDataSize = rawDataSize;
    }

    public Integer getTrainSize() {
        // Convert from Long to Integer
        return (int) (long) trainSize;
    }

    public Integer getTestSize() {
        // Convert from Long to Integer
        return (int) (long) testSize;
    }

    public Integer getNumberOfTimesteps() {
        // Convert from Long to Integer
        return (int) (long) numberOfTimesteps;
    }

    public void setNumberOfTimesteps(Long numberOfTimesteps) {
        this.numberOfTimesteps = numberOfTimesteps;
    }

    public Integer getMiniBatchSize() {
        // Convert from Long to Integer
        return (int) (long) miniBatchSize;
    }

}

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
package se.bigdatamining.forecastmaster.graphs;

import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import static java.time.temporal.ChronoUnit.DAYS;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.PostConstruct;
import javax.enterprise.context.RequestScoped;
import javax.inject.Inject;
import javax.inject.Named;
import org.neo4j.driver.v1.Record;
import org.neo4j.driver.v1.Session;
import org.neo4j.driver.v1.StatementResult;
import org.neo4j.driver.v1.Values;
import org.neo4j.driver.v1.exceptions.ClientException;
import org.primefaces.model.chart.Axis;
import org.primefaces.model.chart.AxisType;
import org.primefaces.model.chart.LineChartModel;
import org.primefaces.model.chart.ChartSeries;
import org.primefaces.model.chart.DateAxis;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.bigdatamining.forecastmaster.Neo4jBean;
import se.bigdatamining.forecastmaster.User;
import se.bigdatamining.forecastmaster.query.QueryBean;

/**
 * This class provides plotting service for the actual loaded data, i.e., it
 * doesn't contain any predicted data.
 *
 * @author Magnus Palm
 */
@Named(value = "dataViewBean")
@RequestScoped // Will refresh on F5
public class DataViewBean {

    @Inject
    User user;

    @Inject
    Neo4jBean neo4jBean;

    @Inject
    QueryBean query;

    private static final Logger LOGGER = LoggerFactory.getLogger(DataViewBean.class);
    private Session session;

    // ADD CLASS SPECIFIC MAPS AND FIELDS HERE
    private Map<LocalDate, ChartData> actDataMap;
    private Map<LocalDate, ChartData> predDataMap;
    private Map<LocalDate, ChartData> fcDataMap;

    private LineChartModel lineModelActual;
    private LineChartModel lineModelPredict;
    private LineChartModel lineModelForecast;

    @PostConstruct
    public void init() {
        // INITIATE CLASS SPECIFIC MAPS AND FIELDS HERE - THE ORDER IS IMPORTANT

        // Initialize driver
        this.session = neo4jBean.getDRIVER().session();

        // Initialize the Actual values Data Map
        this.actDataMap = new LinkedHashMap<>();

        // Initialize the Predicted values Data Map
        this.predDataMap = new LinkedHashMap<>();

        // Initialize the Forecast values Data Map
        this.fcDataMap = new LinkedHashMap<>();

        // Populate Actual values Data Map with data from database
        populateActDataMap();

        // Populate Predicted values Data Map with data from database
        populatePredDataMap();

        // Populate Forecast values Data Map with data from database
        populateFcDataMap();

        // Create the line models
        createLineModels();
    }

    public LineChartModel getLineModelActual() {
        return lineModelActual;
    }

    public LineChartModel getLineModelPredict() {
        return lineModelPredict;
    }

    public LineChartModel getLineModelForecast() {
        return lineModelForecast;
    }

    /**
     * Populates actDataMap with data from database
     *
     */
    public void populateActDataMap() {
        try {

            String tx = "MATCH (p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}) RETURN p.msEpoch AS ms, p.respVar0 AS respVar0 ORDER BY ms";

            StatementResult result = this.session.run(tx, Values.parameters(
                    "custNo", user.getCustomerNumber()
            ));

            while (result.hasNext()) {
                Record next = result.next();

                Long ms = next.get("ms").asLong();
                double respVar0 = next.get("respVar0").asDouble();

                LocalDate date = Instant.ofEpochMilli(ms).atZone(ZoneId.systemDefault()).toLocalDate();

                // Add results to Map
                actDataMap.put(date, new ChartData(date, respVar0));
            }
            LOGGER.info("SUCCESS: Added chart data to actDataMap");
        } catch (ClientException e) {
            LOGGER.error("Exception in 'populateActDataMap' {}", e);
        }
    }

    /**
     * Populates predDataMap with data from database
     */
    private void populatePredDataMap() {
        try {

            String tx = "MATCH (pr:Prediction)-[:PREDICTED_FOR]->(p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}) RETURN p.msEpoch AS ms, pr.predicted AS pred ORDER BY ms";

            StatementResult result = this.session.run(tx, Values.parameters(
                    "custNo", user.getCustomerNumber()
            ));

            while (result.hasNext()) {
                Record next = result.next();

                Long ms = next.get("ms").asLong();
                double predictedValue = next.get("pred").asDouble();

                LocalDate date = Instant.ofEpochMilli(ms).atZone(ZoneId.systemDefault()).toLocalDate();

                // Add results to Map
                predDataMap.put(date, new ChartData(date, predictedValue));
            }
            LOGGER.info("SUCCESS: Added chart data to predDataMap");
        } catch (ClientException e) {
            LOGGER.error("Exception in 'populatePredDataMap' {}", e);
        }
    }

    /**
     * Populates fcDataMap with data from database
     */
    private void populateFcDataMap() {

        // Handle NPE
        if (!actDataMap.isEmpty()) {
            // Calculate the average time period and use that as time periods for t+1, t+2, ..., t+n
            LocalDate lastEntry = Collections.max(actDataMap.keySet());
            LocalDate firstEntry = Collections.min(actDataMap.keySet());
            long daysBetween = DAYS.between(firstEntry, lastEntry);
            long increment = daysBetween / actDataMap.size(); // averagePeriod

            // Call the future time predictions from the Query
            long days = 0L;
            double[] fcArray = query.getForecasts();
            for (int i = 0; i < fcArray.length; i++) {
                double prediction = fcArray[i];
                // Put in the fcDataMap
                days = days + increment;
                LocalDate date = lastEntry.plusDays(days);
                fcDataMap.put(date, new ChartData(date, prediction));
            }
        }
    }

    private void createLineModels() {

        lineModelActual = initCategoryModelAct();
        lineModelActual.setTitle("Historical Data");
        lineModelActual.setLegendPosition("nw");
        lineModelActual.getAxes().put(AxisType.X, new DateAxis("Date"));
        Axis yAxis = lineModelActual.getAxis(AxisType.Y);
        yAxis = lineModelActual.getAxis(AxisType.Y);
        yAxis.setLabel("Value");

        lineModelPredict = initCategoryModelPred();
        lineModelPredict.setTitle("Historical and Predicted Data Fit");
        lineModelPredict.setLegendPosition("nw");
        lineModelPredict.getAxes().put(AxisType.X, new DateAxis("Date"));
        yAxis = lineModelPredict.getAxis(AxisType.Y);
        yAxis = lineModelPredict.getAxis(AxisType.Y);
        yAxis.setLabel("Value");

        lineModelForecast = initCategoryModelFC();
        lineModelForecast.setTitle("Historical and Forecast Data");
        lineModelForecast.setLegendPosition("nw");
        lineModelForecast.getAxes().put(AxisType.X, new DateAxis("Date"));
        yAxis = lineModelForecast.getAxis(AxisType.Y);
        yAxis = lineModelForecast.getAxis(AxisType.Y);
        yAxis.setLabel("Value");
    }

    private LineChartModel initCategoryModelAct() {
        LineChartModel model = new LineChartModel();

        ChartSeries actual = new ChartSeries();

        actDataMap.entrySet().stream().forEach((entry) -> {
            LocalDate key = entry.getKey();
            ChartData value = entry.getValue();

            // Transform date to date format
            String chartDate = key.format(DateTimeFormatter.ISO_DATE);

            // add data to actual data series
            actual.set(chartDate, value.getResponeVar0());
        });

        actual.setLabel("Actual Data");

        // Add the first series
        model.addSeries(actual);
        // Add additional series ...

        return model;
    }

    private LineChartModel initCategoryModelPred() {
        LineChartModel model = new LineChartModel();

        ChartSeries actual = new ChartSeries();
        ChartSeries predicted = new ChartSeries();

        actDataMap.entrySet().stream().forEach((entry) -> {
            LocalDate key = entry.getKey();
            ChartData value = entry.getValue();

            // Transform date to date format
            String chartDate = key.format(DateTimeFormatter.ISO_DATE);

            // add data to actual data series
            actual.set(chartDate, value.getResponeVar0());
        });

        predDataMap.entrySet().stream().forEach((entry) -> {
            LocalDate key = entry.getKey();
            ChartData value = entry.getValue();

            // Transform date to date format
            String chartDate = key.format(DateTimeFormatter.ISO_DATE);

            // add data to predicted data series
            predicted.set(chartDate, value.getResponeVar0() /* This is the predicted value */);
        });

        actual.setLabel("Actual Data");
        predicted.setLabel("Predicted test data");

        // Add the first series
        model.addSeries(actual);
        model.addSeries(predicted);
        // Add additional series ...

        return model;
    }

    private LineChartModel initCategoryModelFC() {
        LineChartModel model = new LineChartModel();

        ChartSeries actual = new ChartSeries();
        ChartSeries forecast = new ChartSeries();

        actDataMap.entrySet().stream().forEach((entry) -> {
            LocalDate key = entry.getKey();
            ChartData value = entry.getValue();

            // Transform date to date format
            String chartDate = key.format(DateTimeFormatter.ISO_DATE);

            // add data to actual data series
            actual.set(chartDate, value.getResponeVar0());
        });

        fcDataMap.entrySet().stream().forEach((entry) -> {
            LocalDate key = entry.getKey();
            ChartData value = entry.getValue();

            // Transform date to date format
            String chartDate = key.format(DateTimeFormatter.ISO_DATE);

            // add data to forecast data series
            forecast.set(chartDate, value.getResponeVar0() /* This is the forecast value */);
        });

        actual.setLabel("Actual Data");
        forecast.setLabel("Forecast");

        // Add the first series
        model.addSeries(actual);
        model.addSeries(forecast);
        // Add additional series ...

        return model;
    }

}

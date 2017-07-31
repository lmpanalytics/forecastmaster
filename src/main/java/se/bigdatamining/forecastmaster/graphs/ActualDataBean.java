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

import java.io.Serializable;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
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

/**
 * This class provides plotting service for the actual loaded data, i.e., it
 * doesn't contain any predicted data.
 *
 * @author Magnus Palm
 */
@Named(value = "actualDataBean")
@RequestScoped // Will refresh on F5
public class ActualDataBean implements Serializable {

    @Inject
    Neo4jBean neo4jBean;

    private static final long serialVersionUID = 1L;
    private static final Logger LOGGER = LoggerFactory.getLogger(ActualDataBean.class);
    private Session session;

    // ADD CLASS SPECIFIC MAPS AND FIELDS HERE
    private Map<LocalDate, ChartData> actDataMap;

    private LineChartModel lineModel;

    @PostConstruct
    public void init() {
        // INITIATE CLASS SPECIFIC MAPS AND FIELDS HERE - THE ORDER IS IMPORTANT

        // Initialize driver
        this.session = neo4jBean.getDRIVER().session();

        // Initialize the actDataMap
        this.actDataMap = new LinkedHashMap<>();

        // Populate actDataMap with data from database
        populateActDataMap();

        // Create the line models
        createLineModels();
    }

    public LineChartModel getLineModel() {
        return lineModel;
    }

    /**
     * Populates actDataMap with data from database
     *
     */
    public void populateActDataMap() {
        try {

            String tx = "MATCH (p:Pattern)-[:OWNED_BY]->(c:Customer {customerNumber:$custNo}) RETURN p.msEpoch AS ms, p.respVar0 AS respVar0 ORDER BY ms";

            StatementResult result = this.session.run(tx, Values.parameters(
                    "custNo", "0000000001"
            ));

            while (result.hasNext()) {
                Record next = result.next();

                Long ms = next.get("ms").asLong();
                double respVar0 = next.get("respVar0").asDouble();

                LocalDate date = Instant.ofEpochMilli(ms).atZone(ZoneId.systemDefault()).toLocalDate();

                // Add results to Map
                actDataMap.put(date, new ChartData(date, respVar0));
            }
            LOGGER.info("SUCCESS: Added chart data to ActDataMap");
        } catch (ClientException e) {
            LOGGER.error("Exception in 'populateActDataMap' {}", e);
        }
    }

    private void createLineModels() {

        lineModel = initCategoryModel();
        lineModel.setTitle("Historical Data");
        lineModel.setLegendPosition("nw");
        lineModel.getAxes().put(AxisType.X, new DateAxis("Date"));
        Axis yAxis = lineModel.getAxis(AxisType.Y);
        yAxis = lineModel.getAxis(AxisType.Y);
        yAxis.setLabel("Value");
    }

    private LineChartModel initCategoryModel() {
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

}

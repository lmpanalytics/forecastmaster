/*
 * Copyright 2017 Magnus Palm.
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
package se.bigdatamining.forecastmaster;

/**
 * This class writes and reads data from an Excel file to a Neo4j DB
 *
 * @author Magnus Palm
 */
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import javax.annotation.security.DeclareRoles;
import javax.annotation.security.RolesAllowed;
import javax.ejb.Stateful;
import javax.enterprise.context.SessionScoped;
import javax.faces.application.FacesMessage;
import javax.faces.context.FacesContext;
import javax.inject.Inject;
import javax.inject.Named;
import org.apache.poi.hssf.usermodel.HSSFDateUtil;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.neo4j.driver.v1.Session;
import org.neo4j.driver.v1.Transaction;
import static org.neo4j.driver.v1.Values.parameters;

import org.primefaces.event.FileUploadEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Named(value = "fileLoadView")
@SessionScoped
@Stateful
@DeclareRoles({"BASIC", "PRO", "DIRECTOR", "MDM", "FORECASTER", "TRAINER"})
public class FileLoadView {

    @Inject
    Neo4jBean neo;

    @Inject
    User user;

    private static final Logger LOGGER = LoggerFactory.getLogger(FileLoadView.class);
    private static FileInputStream excelFile;

    /**
     * Handles the file upload
     *
     * @param event
     */
    @RolesAllowed({"BASIC", "PRO", "DIRECTOR", "MDM"})
    public void handleFileUpload(FileUploadEvent event) {

        try {
            FacesMessage message = new FacesMessage("Succesful", event.getFile().getFileName() + " is uploaded.");
            FacesContext.getCurrentInstance().addMessage(null, message);

//            Access the uploaded file from memory
            this.excelFile = (FileInputStream) event.getFile().getInputstream();

//        Convert the file to Excel format and populate pattern map
            Map<Long, Pattern> patternMap = readExcel();

//        Write data patterns to the data base
            addPattern(patternMap);

            LOGGER.info("Data written to DB.");
        } catch (IOException ex) {
            LOGGER.error("Exception in handleFileUpload method {}", ex);
        }
    }

    /**
     * Reads an Excel file, based on:
     * https://www.mkyong.com/java/apache-poi-reading-and-writing-excel-file-in-java/
     */
    private Map<Long, Pattern> readExcel() {
        Map<Long, Pattern> m = new HashMap<>();

        try {

            Workbook workbook = new XSSFWorkbook(excelFile);
            Sheet datatypeSheet = workbook.getSheetAt(0);
            Iterator<Row> rowIterator = datatypeSheet.iterator();

            while (rowIterator.hasNext()) {

                Row currentRow = rowIterator.next();
                Iterator<Cell> cellIterator = currentRow.iterator();

//                Initiate the date time (in ms since 1970-01-01) and the response variable
                Long msTime = Long.MAX_VALUE;
                Double responeVar = 0d;

                while (cellIterator.hasNext()) {

                    Cell currentCell = cellIterator.next();
                    //getCellTypeEnum shown as deprecated for version 3.15
                    //getCellTypeEnum ill be renamed to getCellType starting from version 4.0
                    if (currentCell.getCellTypeEnum() == CellType.STRING) {
//                        System.out.print(currentCell.getStringCellValue() + " I'm a String ");
//  HSSFDateUtil will throw an exception if fed a String, hence placed after above condition
                    } else if (HSSFDateUtil.isCellInternalDateFormatted(currentCell)) {
                        msTime = currentCell.getDateCellValue().getTime();
//                        System.out.print(msTime + " I'm a date ");
                    } else if (currentCell.getCellTypeEnum() == CellType.NUMERIC) {
                        responeVar = currentCell.getNumericCellValue();
//                        System.out.print(data + " I'm a number ");
                    } else if (currentCell.getCellTypeEnum() == CellType.FORMULA) {
//                        System.out.print(currentCell.getNumericCellValue() + " I'm a formula ");
                    }

//                    Add to map if and only if the Excel row starts with a valid internal date format
                    if (msTime != Long.MAX_VALUE) {
                        m.put(msTime, new Pattern(msTime, responeVar));
                    }
                }
            }

        } catch (FileNotFoundException e) {
        } catch (IOException e) {
        }
        return m;
    }

    /**
     * Adds data patterns to the data base
     *
     * @param pattern
     */
    private void addPattern(Map<Long, Pattern> pattern) {
        // Sessions are lightweight and disposable connection wrappers.
        try (Session session = neo.getDRIVER().session()) {
            // Wrapping Cypher in an explicit transaction provides atomicity
            // and makes handling errors much easier.
            try (Transaction tx = session.beginTransaction()) {
                pattern.values().stream().forEach((v) -> {
                    Double respVar0 = v.getResponeVar0();
                    Long ms = v.getMsTime();
                    tx.run("MATCH (c:Customer { customerNumber: {custNo} })"
                            + "MERGE (p:Pattern {msEpoch: {t}, respVar0: {rv0}}) "
                            + "MERGE (p)-[:OWNED_BY]->(c)",
                            parameters("t", ms,
                                    "rv0", respVar0,
                                    "custNo", user.getCustomerNumber()));
                    tx.success();  // Mark this write as successful.
                });
            }
        }
    }

}

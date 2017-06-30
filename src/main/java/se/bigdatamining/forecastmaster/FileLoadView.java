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
 *
 * @author Magnus Palm
 */
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.util.Iterator;
import javax.enterprise.context.SessionScoped;
import javax.faces.application.FacesMessage;
import javax.faces.context.FacesContext;
import javax.inject.Named;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import org.primefaces.event.FileUploadEvent;

@Named(value = "fileLoadView")
@SessionScoped
public class FileLoadView implements Serializable {

    private static final long serialVersionUID = 1L;
    private static FileInputStream excelFile;

    /**
     * Handles the file upload
     *
     * @param event
     * @throws Exception
     */
    public void handleFileUpload(FileUploadEvent event) throws Exception {

        FacesMessage message = new FacesMessage("Succesful", event.getFile().getFileName() + " is uploaded.");
        FacesContext.getCurrentInstance().addMessage(null, message);

//            Access the uploaded file from memory
        this.excelFile = (FileInputStream) event.getFile().getInputstream();

//        Convert the file to Excel format
        readExcel();
    }

    /**
     * Reads an Excel file, based on:
     * https://www.mkyong.com/java/apache-poi-reading-and-writing-excel-file-in-java/
     */
    private void readExcel() {
        try {

            Workbook workbook = new XSSFWorkbook(this.excelFile);
            Sheet datatypeSheet = workbook.getSheetAt(0);
            Iterator<Row> iterator = datatypeSheet.iterator();

            while (iterator.hasNext()) {

                Row currentRow = iterator.next();
                Iterator<Cell> cellIterator = currentRow.iterator();

                while (cellIterator.hasNext()) {

                    Cell currentCell = cellIterator.next();
                    //getCellTypeEnum shown as deprecated for version 3.15
                    //getCellTypeEnum ill be renamed to getCellType starting from version 4.0
                    if (currentCell.getCellTypeEnum() == CellType.STRING) {
                        System.out.print(currentCell.getStringCellValue() + "--");
                    } else if (currentCell.getCellTypeEnum() == CellType.NUMERIC) {
                        System.out.print(currentCell.getNumericCellValue() + "--");
                    } else if (currentCell.getCellTypeEnum() == CellType.FORMULA) {
                        System.out.print(currentCell.getNumericCellValue() + "-*-");
                    }

                }
//                System.out.println();

            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

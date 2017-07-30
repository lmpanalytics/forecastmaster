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

import java.time.LocalDate;

/**
 *
 * This class models a chart data object.
 *
 * @author Magnus Palm
 */
public class ChartData {

    private LocalDate date;
    private Double responeVar0;

    /**
     * Constructor for ChartData Objects
     *
     * @param date Transaction date
     * @param responeVar0 Response variable 1
     */
    public ChartData(LocalDate date, Double responeVar0) {
        this.date = date;
        this.responeVar0 = responeVar0;
    }

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
        this.date = date;
    }

    public Double getResponeVar0() {
        return responeVar0;
    }

    public void setResponeVar0(Double responeVar0) {
        this.responeVar0 = responeVar0;
    }

}

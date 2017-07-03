/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package se.bigdatamining.forecastmaster;

/**
 * This class models data patterns e.g., sales transactions
 *
 * @author Magnus
 */
public class Pattern {

    private Long msTime /* time in ms since 1970-01-01 */;
    private Double responeVar0 /* Response variable 1 */;

    public Pattern(Long msTime, Double responeVar0) {
        this.msTime = msTime;
        this.responeVar0 = responeVar0;
    }

    public Long getMsTime() {
        return msTime;
    }

    public void setMsTime(Long msTime) {
        this.msTime = msTime;
    }

    public Double getResponeVar0() {
        return responeVar0;
    }

    public void setResponeVar0(Double responeVar0) {
        this.responeVar0 = responeVar0;
    }

}

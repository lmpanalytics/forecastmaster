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
package se.bigdatamining.forecastmaster;

import java.sql.Timestamp;
import javax.annotation.PreDestroy;
import javax.ejb.Singleton;
import org.neo4j.driver.v1.AuthTokens;
import org.neo4j.driver.v1.Driver;
import org.neo4j.driver.v1.GraphDatabase;

/**
 * This bean provides the database service
 *
 * @author Magnus Palm
 */
@Singleton
public class Neo4jBean {

    private static final String URI = "bolt://localhost:7687";
    private static final String USER = "neo4j";
    private static final String PASSWORD = "Tokyo2000";

    private final Driver DRIVER;
    private Timestamp timestamp;

    public Neo4jBean() {
        this.DRIVER = GraphDatabase.driver(URI, AuthTokens.basic(USER, PASSWORD));
    }

    @PreDestroy
    public void destroyMe() {
        DRIVER.session().close();
        DRIVER.close();
        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println(timestamp + ": Neo4jDriver in Neo4jBean has been disposed of.");
    }

    /**
     *
     * @return the DB driver
     */
    public Driver getDRIVER() {
        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println(timestamp + ": Aquire Neo4jDriver.");
        return DRIVER;
    }

    /**
     * Close the DB driver
     */
    public void closeNeo4jDriver() {
        DRIVER.close();
        timestamp = new Timestamp(System.currentTimeMillis());
        System.out.println(timestamp + ": closed Neo4jDriver.");
    }
}

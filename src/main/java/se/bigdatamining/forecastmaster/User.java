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

import javax.inject.Named;
import javax.enterprise.context.SessionScoped;
import java.io.Serializable;
import javax.annotation.PostConstruct;
import javax.annotation.Resource;
import javax.ejb.SessionContext;
import javax.ejb.Startup;
import javax.ejb.Stateful;
import javax.inject.Inject;
import org.neo4j.driver.v1.Session;
import org.neo4j.driver.v1.StatementResult;
import org.neo4j.driver.v1.Transaction;
import static org.neo4j.driver.v1.Values.parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Matches the logged in session user with the customer number. This is possible
 * as the user names and customer numbers are constrained in the database to be
 * unique.
 *
 * @author Magnus Palm
 */
@Named(value = "user")
@Startup
@SessionScoped
@Stateful
public class User implements Serializable{

    @Inject
    Neo4jBean neo;

    private String userName;
    private String customerNumber;

    @Resource
    SessionContext ctx;

    private static Session session;
    private static final Logger LOGGER = LoggerFactory.getLogger(User.class);

    /**
     * Creates a new instance of User
     */
    public User() {

    }

    @PostConstruct
    public void init() {

        // Initialize driver
        session = neo.getDRIVER().session();

        // Handle NPE
        if (ctx.getCallerPrincipal() != null) {
            userName = ctx.getCallerPrincipal().getName();
        }
        // Handle NPE
        if (userName != null) {
            customerNumber = queryCustomerNumber(userName);
            LOGGER.info("SUCCESS: Matching user {} to customer {}", userName, customerNumber);
        } else {
            LOGGER.error("Matching user {} to customer", userName);
        }

    }

    public static String queryCustomerNumber(String userName) {
        return session.readTransaction((Transaction tx) -> matchCustomerNode(tx, userName));
    }

    private static String matchCustomerNode(Transaction tx, String userName) {
        StatementResult result = tx.run("MATCH (u:User {userName: $name})-[:AT_CUSTOMER]-(c:Customer) RETURN c.customerNumber AS custNumber", parameters("name", userName));
        return result.single().get(0).asString();
    }

    public String getUserName() {
        return userName;
    }

    public String getCustomerNumber() {
        return customerNumber;
    }

}

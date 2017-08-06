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
package se.bigdatamining.forecastmaster.login;

import javax.inject.Named;
import javax.enterprise.context.SessionScoped;
import java.io.Serializable;
import javax.faces.context.FacesContext;
import javax.servlet.http.HttpSession;

/**
 * This bean provides user logout.
 *
 * @author Magnus Palm
 */
@Named(value = "userSession")
@SessionScoped
public class LogOutBean implements Serializable {

    /**
     * Creates a new instance of LogOutBean
     */
    public LogOutBean() {
    }

    public String logout() {
        HttpSession session = (HttpSession) FacesContext.getCurrentInstance().getExternalContext().getSession(true);
        session.invalidate();
        return "/index?faces-redirect=true";
    }

    public boolean isUserLoggedIn() {
        String user = this.getUsername();
        boolean result = !((user == null) || user.isEmpty());
        return result;
    }

    /**
     * Get the login username if it exists
     */
    public String getUsername() {
        String user = FacesContext.getCurrentInstance().getExternalContext().getRemoteUser();
        return user;
    }
}

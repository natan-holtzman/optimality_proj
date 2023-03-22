# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:01:48 2023

@author: natan
"""


import numpy as np
import scipy.optimize

def fit_gpp_lai_evi(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    nrad_arr = np.array(dfgpp.normrad)
    evi_arr = np.array(dfgpp.EVI2)/np.max(dfgpp.EVI2)
    lai_arr = np.array(dfgpp.LAI)/np.max(dfgpp.LAI)

    day_effect = np.array(dfgpp.dayfrac)

    def tofit(pars):
        t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
        t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
        
        t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
        t_effect2 /= np.exp(-(25-pars[3])**2/20**2/2)
        
        par_effect = (nrad_arr/275)**max(0,pars[4])
        lai_weight = max(0,min(1,pars[5]))
        leaf_effect = lai_weight*lai_arr + (1-lai_weight)*evi_arr
        gppmax = max(0,pars[0]) * t_effect1 * par_effect* leaf_effect * day_effect
        slope = max(0,pars[1]) * t_effect2
        gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        return gpp_pred - gpp_arr
    fit0 = np.array([np.max(gpp_arr),120,25,25,0.5,0.5])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
    t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
    
    t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
    t_effect2 /= np.exp(-(25-pars[3])**2/20**2/2)
    
    par_effect = (nrad_arr/275)**max(0,pars[4])
    lai_weight = max(0,min(1,pars[5]))
    leaf_effect = lai_weight*lai_arr + (1-lai_weight)*evi_arr
    gppmax = max(0,pars[0]) * t_effect1 * par_effect* leaf_effect * day_effect
    slope = max(0,pars[1]) * t_effect2
    gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        
    dfgpp["gppmax"] = gppmax
    dfgpp["gppbase"] = max(0,pars[0])
    dfgpp["slope0"] = max(0,pars[1])
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
    dfgpp["emax"] = pars[2]
    dfgpp["eslope"] = pars[3]
    dfgpp["parfac"] = max(0,pars[4])
    dfgpp["laiweight"] = max(0,min(1,pars[5]))
    
    return dfgpp
#%%

def fit_gpp_lai_only(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    nrad_arr = np.array(dfgpp.normrad)
    lai_arr = np.array(dfgpp.LAI)/np.max(dfgpp.LAI)

    day_effect = np.array(dfgpp.dayfrac)

    def tofit(pars):
        t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
        t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
        
        t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
        t_effect2 /= np.exp(-(25-pars[3])**2/20**2/2)
        
        par_effect = (nrad_arr/275)**np.exp(pars[4])
        leaf_effect = 1*lai_arr
        gppmax = np.exp(pars[0]) * t_effect1 * par_effect* leaf_effect * day_effect
        slope = np.exp(pars[1]) * t_effect2
        gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        return gpp_pred - gpp_arr
    fit0 = np.array([np.log(np.max(gpp_arr)),np.log(120),25,25,np.log(0.5)])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
    t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
    
    t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
    t_effect2 /= np.exp(-(25-pars[3])**2/20**2/2)
    
    par_effect = (nrad_arr/275)**np.exp(pars[4])
    leaf_effect = 1*lai_arr
    gppmax = np.exp(pars[0]) * t_effect1 * par_effect* leaf_effect * day_effect
    slope = np.exp(pars[1]) * t_effect2
    gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        
    dfgpp["gppmax"] = gppmax
    dfgpp["gppbase"] = np.exp(pars[0])
    dfgpp["slope0"] = np.exp(pars[1])
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
    dfgpp["emax"] = pars[2]
    dfgpp["eslope"] = pars[3]
    dfgpp["parfac"] = np.exp(pars[4])
    #dfgpp["laiweight"] = max(0,min(1,pars[5]))
    
    return dfgpp
#%%

def fit_gpp_linear(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    nrad_arr = np.array(dfgpp.normrad)
    lai_arr = np.array(dfgpp.LAI)/np.max(dfgpp.LAI)

    day_effect = np.array(dfgpp.dayfrac)

    def tofit(pars):
        t_effect1 = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
        t_effect1 /= np.exp(-(25-pars[1])**2/20**2/2)
        
        gpp_pred = t_effect1*cond_arr*pars[0]
        return gpp_pred - gpp_arr
    fit0 = np.array([np.median(gpp_arr/cond_arr),25])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    t_effect1 = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
    t_effect1 /= np.exp(-(25-pars[1])**2/20**2/2)
    
    gpp_pred = t_effect1*cond_arr*pars[0]
    
    dfgpp["gpp_pred"] = gpp_pred

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    return dfgpp
#%%

def fit_gpp_lai_slope(df1,slope0):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    nrad_arr = np.array(dfgpp.normrad)
    lai_arr = np.array(dfgpp.LAI)/np.max(dfgpp.LAI)

    day_effect = np.array(dfgpp.dayfrac)

    def tofit(pars):
        t_effect1 = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
        t_effect1 /= np.exp(-(25-pars[1])**2/20**2/2)
        
        t_effect2 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
        t_effect2 /= np.exp(-(25-pars[2])**2/20**2/2)
        
        par_effect = (nrad_arr/275)**np.exp(pars[3])
        leaf_effect = 1*lai_arr
        gppmax = np.exp(pars[0]) * t_effect1 * par_effect* leaf_effect * day_effect
        slope = slope0 * t_effect2
        gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        return gpp_pred - gpp_arr
    fit0 = np.array([np.log(np.max(gpp_arr)),25,25,np.log(0.5)])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
     
    t_effect1 = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
    t_effect1 /= np.exp(-(25-pars[1])**2/20**2/2)
    
    t_effect2 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
    t_effect2 /= np.exp(-(25-pars[2])**2/20**2/2)
    
    par_effect = (nrad_arr/275)**np.exp(pars[3])
    leaf_effect = 1*lai_arr
    gppmax = np.exp(pars[0]) * t_effect1 * par_effect* leaf_effect * day_effect
    slope = slope0 * t_effect2
    gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        
    dfgpp["gppmax"] = gppmax
    dfgpp["gppbase"] = np.exp(pars[0])
    #dfgpp["slope0"] = np.exp(pars[1])
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
    dfgpp["emax"] = pars[1]
    dfgpp["eslope"] = pars[2]
    dfgpp["parfac"] = np.exp(pars[3])
    #dfgpp["laiweight"] = max(0,min(1,pars[5]))
    
    return dfgpp
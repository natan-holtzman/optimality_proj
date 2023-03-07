# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 08:41:57 2023

@author: natan
"""


import numpy as np
#import pymc as pm
#import pandas as pd
import scipy.optimize
#import statsmodels.formula.api as smf
#%%

def fit_gpp_flex_slope(df_in):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    #slope0 = 97
    def gpp_opt(pars):
        slope0 = pars[3]
        gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[4]
        gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.1,0.1,100,0.5])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    slope0 = pars[3]
    gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[4]
    gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
    dfgpp["slope0"] = slope0
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    dfgpp["par_coef"] = pars[4]
    
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gppmax*(1-np.exp(-cond_arr/gppmax*slope0))
    
    return dfgpp
#%%

def fit_gpp_flex_slope_TP(df_in):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    #slope0 = 97
    def gpp_opt(pars):
        slope_base = pars[1]
        max0 = pars[0]
        par_coef = max(0,pars[2])
        par_effect = (par_arr/275)**par_coef
        t_eff_max = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
        t_eff_max /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
        t_eff_slope = np.exp(-(t_arr-pars[4])**2 / 20**2 / 2)
        t_eff_slope /= np.exp(-(25-pars[4])**2 / 20**2 / 2)

        gppmax = max0*t_eff_max*par_effect
        slope0 = slope_base*t_eff_slope
        gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,110,0.5,25,30])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    slope_base = pars[1]
    max0 = pars[0]
    par_coef = max(0,pars[2])
    par_effect = (par_arr/275)**par_coef
    t_eff_max = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
    t_eff_max /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
    t_eff_slope = np.exp(-(t_arr-pars[4])**2 / 20**2 / 2)
    t_eff_slope /= np.exp(-(25-pars[4])**2 / 20**2 / 2)

    gppmax = max0*t_eff_max*par_effect
    slope0 = slope_base*t_eff_slope
    gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
    dfgpp["slope0"] = slope_base
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    #dfgpp["par_coef"] = pars[4]
    
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gpp_pred
    
    return dfgpp
#%%

def fit_gpp_flex_slope_TP_season(df_in):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    #slope0 = 97
    def gpp_opt(pars):
        slope_base = pars[1]
        max0 = pars[0]
        par_coef = max(0,pars[2])
        par_effect = (par_arr/275)**par_coef
        t_eff_max = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
        t_eff_max /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
        t_eff_slope = np.exp(-(t_arr-pars[4])**2 / 20**2 / 2)
        t_eff_slope /= np.exp(-(25-pars[4])**2 / 20**2 / 2)
        season_effect = (1 - max(0,pars[5])*spring_arr - max(0,pars[6])*fall_arr)
        gppmax = max0*t_eff_max*par_effect*season_effect
        slope0 = slope_base*t_eff_slope
        gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,110,0.5,25,30,0.1,0.1])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    slope_base = pars[1]
    max0 = pars[0]
    par_coef = max(0,pars[2])
    par_effect = (par_arr/275)**par_coef
    t_eff_max = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
    t_eff_max /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
    t_eff_slope = np.exp(-(t_arr-pars[4])**2 / 20**2 / 2)
    t_eff_slope /= np.exp(-(25-pars[4])**2 / 20**2 / 2)
    season_effect = (1 - max(0,pars[5])*spring_arr - max(0,pars[6])*fall_arr)
    gppmax = max0*t_eff_max*par_effect*season_effect
    slope0 = slope_base*t_eff_slope
    gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
    dfgpp["slope0"] = slope_base
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    #dfgpp["par_coef"] = pars[4]
    
    dfgpp["par_coef"] = par_coef
    dfgpp["spring_coef"] = max(0,pars[5])
    dfgpp["fall_coef"] = max(0,pars[6])
    dfgpp["topt_gmax"] = pars[3]
    dfgpp["topt_slope"] = pars[4]
    dfgpp["max0"] = max0
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gpp_pred
    
    return dfgpp
#%%

def fit_gpp_flex_slope_TP_season_res(df_in):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    #slope0 = 97
    def gpp_opt(pars):
        slope_base = pars[1]
        max0 = pars[0]
        par_coef = max(0,pars[2])
        par_effect = (par_arr/275)**par_coef
        t_eff_max = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
        t_eff_max /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
        t_eff_slope = np.exp(-(t_arr-pars[4])**2 / 20**2 / 2)
        t_eff_slope /= np.exp(-(25-pars[4])**2 / 20**2 / 2)
        season_effect = (1 - max(0,pars[5])*spring_arr - max(0,pars[6])*fall_arr)
        gppmax = max0*t_eff_max*par_effect*season_effect
        slope0 = slope_base*t_eff_slope
        cond_res = max(0,pars[7])
        cond_plant = np.clip(cond_arr-cond_res,0,np.inf)
        gpp_pred = gppmax*(1-np.exp(-cond_plant*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,110,0.5,25,30,0.1,0.1,0.001])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    slope_base = pars[1]
    max0 = pars[0]
    par_coef = max(0,pars[2])
    par_effect = (par_arr/275)**par_coef
    t_eff_max = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
    t_eff_max /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
    t_eff_slope = np.exp(-(t_arr-pars[4])**2 / 20**2 / 2)
    t_eff_slope /= np.exp(-(25-pars[4])**2 / 20**2 / 2)
    season_effect = (1 - max(0,pars[5])*spring_arr - max(0,pars[6])*fall_arr)
    gppmax = max0*t_eff_max*par_effect*season_effect
    slope0 = slope_base*t_eff_slope
    cond_res = max(0,pars[7])
    cond_plant = np.clip(cond_arr-cond_res,0,np.inf)
    gpp_pred = gppmax*(1-np.exp(-cond_plant*slope0/gppmax))
    
    dfgpp["slope0"] = slope_base
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    #dfgpp["par_coef"] = pars[4]
    
    dfgpp["par_coef"] = par_coef
    dfgpp["spring_coef"] = max(0,pars[5])
    dfgpp["fall_coef"] = max(0,pars[6])
    dfgpp["topt_gmax"] = pars[3]
    dfgpp["topt_slope"] = pars[4]
    dfgpp["max0"] = max0
    dfgpp["res_cond"] = cond_res

    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gpp_pred
    
    return dfgpp
#%%

def fit_gpp_fix_slope_TP_season_res(df_in,slope_base):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    #slope0 = 97
    def gpp_opt(pars):
        max0 = pars[0]
        par_coef = max(0,pars[1])
        par_effect = (par_arr/275)**par_coef
        t_eff_max = np.exp(-(t_arr-pars[2])**2 / 20**2 / 2)
        t_eff_max /= np.exp(-(25-pars[2])**2 / 20**2 / 2)
        t_eff_slope = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
        t_eff_slope /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
        season_effect = (1 - max(0,pars[4])*spring_arr - max(0,pars[5])*fall_arr)
        gppmax = max0*t_eff_max*par_effect*season_effect
        slope0 = slope_base*t_eff_slope
        cond_res = max(0,pars[6])
        cond_plant = np.clip(cond_arr-cond_res,0,np.inf)
        gpp_pred = gppmax*(1-np.exp(-cond_plant*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.5,25,30,0.1,0.1,0.001])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    max0 = pars[0]
    par_coef = max(0,pars[1])
    par_effect = (par_arr/275)**par_coef
    t_eff_max = np.exp(-(t_arr-pars[2])**2 / 20**2 / 2)
    t_eff_max /= np.exp(-(25-pars[2])**2 / 20**2 / 2)
    t_eff_slope = np.exp(-(t_arr-pars[3])**2 / 20**2 / 2)
    t_eff_slope /= np.exp(-(25-pars[3])**2 / 20**2 / 2)
    season_effect = (1 - max(0,pars[4])*spring_arr - max(0,pars[5])*fall_arr)
    gppmax = max0*t_eff_max*par_effect*season_effect
    slope0 = slope_base*t_eff_slope
    cond_res = max(0,pars[6])
    cond_plant = np.clip(cond_arr-cond_res,0,np.inf)
    gpp_pred = gppmax*(1-np.exp(-cond_plant*slope0/gppmax))
    
    dfgpp["slope0"] = slope_base
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    #dfgpp["par_coef"] = pars[4]
    
    dfgpp["par_coef"] = par_coef
    dfgpp["spring_coef"] = max(0,pars[4])
    dfgpp["fall_coef"] = max(0,pars[5])
    dfgpp["topt_gmax"] = pars[2]
    dfgpp["topt_slope"] = pars[3]
    dfgpp["max0"] = max0
    dfgpp["res_cond"] = cond_res

    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gpp_pred
    
    return dfgpp
#%%
def fit_gpp_fix_slope(df_in,slope0):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    def gpp_opt(pars):
        gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[3]
        gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.1,0.1,0.5])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[3]
    gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
    dfgpp["slope0"] = slope0
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    dfgpp["par_coef"] = pars[3]
    
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gppmax*(1-np.exp(-cond_arr/gppmax*slope0))
    
    return dfgpp
#%%
 
def fit_gpp_fix_slope_res(df_in,slope0):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    def gpp_opt(pars):
        gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[3]
        cond_res = max(0,pars[4])
        cond_plant = np.clip(cond_arr-cond_res,0,np.inf)
        gpp_pred = gppmax*(1-np.exp(-cond_plant*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.1,0.1,0.5,0.001])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[3]
    cond_res = max(0,pars[4])
    cond_plant = np.clip(cond_arr-cond_res,0,np.inf)    
    gpp_pred = gppmax*(1-np.exp(-cond_plant*slope0/gppmax))

    dfgpp["slope0"] = slope0
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    dfgpp["par_coef"] = pars[3]
    dfgpp["res_cond"] = cond_res
    
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gpp_pred
    
    return dfgpp

#%%

def fit_gpp_flex_slope_res(df_in,slope0):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    def gpp_opt(pars):
        gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[3]
        cond_res = max(0,pars[4])
        cond_plant = np.clip(cond_arr-cond_res,0,np.inf)
        gpp_pred = gppmax*(1-np.exp(-cond_plant*pars[5]/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.1,0.1,0.5,0.001,slope0])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**pars[3]
    cond_res = max(0,pars[4])
    cond_plant = np.clip(cond_arr-cond_res,0,np.inf)    
    gpp_pred = gppmax*(1-np.exp(-cond_plant*pars[5]/gppmax))

    dfgpp["slope0"] = pars[5]
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    dfgpp["par_coef"] = pars[3]
    dfgpp["res_cond"] = cond_res
    
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gpp_pred
    
    return dfgpp

#%%
def fit_gpp_fix_slope_day(df_in,slope0):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    day_arr = np.array(dfgpp.dayfrac)
    day_arr /= np.max(day_arr)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    def gpp_opt(pars):
        gppmax = pars[0] * (par_arr/275)**pars[1] * day_arr
        gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.5])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    gppmax = pars[0] * (par_arr/275)**pars[1] * day_arr
    #gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
    dfgpp["slope0"] = slope0
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    dfgpp["par_coef"] = pars[1]
    
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gppmax*(1-np.exp(-cond_arr/gppmax*slope0))
    
    return dfgpp
#%%

def fit_gpp_flex_slope_day(df_in):
    dfgpp = df_in.copy()
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    day_arr = np.array(dfgpp.dayfrac)*np.array(dfgpp.LAIint_rel)
    day_arr /= np.max(day_arr)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    def gpp_opt(pars):
        slope0 = pars[2]
        gppmax = pars[0] * (par_arr/275)**pars[1] * day_arr
        gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.5,100])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    slope0 = pars[2]
    gppmax = pars[0] * (par_arr/275)**pars[1] * day_arr
    #gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
    dfgpp["slope0"] = slope0
    dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    dfgpp["par_coef"] = pars[1]
    
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gppmax*(1-np.exp(-cond_arr/gppmax*slope0))
    
    return dfgpp
    #%%
#     slope_base = 100
#     tmean_points = np.mean(dfgpp.airt)
#     def gpp_opt(pars):
# #        slope0 = pars[4] * np.exp(-(t_arr-pars[5])**2 / 13**2 / 2)

#         #slope0 = pars[4]
#         slope0 =pars[5] 
#         gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[3]*spring_arr) * (par_arr/275)**pars[2] * np.exp(-(t_arr-pars[4])**2 / 15**2 / 2)
#         gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
#         return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
#     fit0 = np.array([gpp_allmax,0.1,0.5,0.1,25,100])
#     gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
#     pars = gpp_optres.x
#     #slope0 = pars[4] 
#     slope0 =pars[5] #* np.exp(-(t_arr-pars[4])**2 / 15**2 / 2)

#     gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[3]*spring_arr) * (par_arr/275)**pars[2] * np.exp(-(t_arr-pars[4])**2 / 15**2 / 2)
#     gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
#     dfgpp["slope_coef"] = pars[5]
#     dfgpp["tcoef"] = pars[4]

#     dfgpp["tmean_points"] = np.mean(dfgpp.airt)
    #%%
    # slope0 = 100#*np.exp(-(t_arr-25)**2 / 20**2 / 2)
    # def gpp_opt(pars):
    #     gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**0.28
    #     gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    #     return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    # fit0 = np.array([gpp_allmax,0.1,0.1])
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    # pars = gpp_optres.x
    # gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[2]*spring_arr) * (par_arr/275)**0.28
    # gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    #dfgpp["par_coef"] = pars[2]
    #%%
    # slope0 = 100
    # def gpp_opt(pars):
    #     gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[3]*spring_arr) * (par_arr/275)**pars[2]
    #     rcond = max(0,pars[4])
    #     adjcond = np.clip(cond_arr-rcond,0,np.inf)
        
    #     gpp_pred = gppmax*(1-np.exp(-adjcond*slope0/gppmax))
    #     return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    # fit0 = np.array([gpp_allmax,0.1,0.5,0.1,gpp_allmax*0.01])
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    # pars = gpp_optres.x
    # gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[3]*spring_arr) * (par_arr/275)**pars[2]
    # rcond = max(0,pars[4])
    # adjcond = np.clip(cond_arr-rcond,0,np.inf)
    # gpp_pred = gppmax*(1-np.exp(-adjcond*slope0/gppmax))
    
    # dfgpp["res_cond"] = rcond
    
    #%%
    # slope0 = 100
    # def gpp_opt(pars):
    #     gppmax = pars[0] * (par_arr/275)**pars[1]
    #     gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    #     return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    # fit0 = np.array([gpp_allmax,0.5])
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    # pars = gpp_optres.x
    # gppmax = pars[0] * (par_arr/275)**pars[1]
    # gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
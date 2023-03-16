# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:43:23 2023

@author: natan
"""
import numpy as np
import scipy.optimize

def fit_gpp(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    Ngpp_arr = np.array(dfgpp.normgpp)
    Ncond_arr = np.array(dfgpp.normcond)
    nrad_arr = np.array(dfgpp.normrad)
    
    def tofit(pars):
        basemax = pars[0]
        slope = pars[1]
        t_effect = np.exp(-(airt_arr-pars[2])**2/20**2/2)
        t_effect /= np.exp(-(25-pars[2])**2/20**2/2)
        par_effect = (nrad_arr/275)**max(0,pars[3])
        gppmax = t_effect*basemax*par_effect
        gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),120,25,0.5])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    basemax = pars[0]
    slope = pars[1]
    t_effect = np.exp(-(airt_arr-pars[2])**2/20**2/2)
    t_effect /= np.exp(-(25-pars[2])**2/20**2/2)
    par_effect = (nrad_arr/275)**max(0,pars[3])
    gppmax = t_effect*basemax*par_effect
    gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
    
    dfgpp["gppmax"] = gppmax*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
#%%
    
    def tofit(pars):
        basemax = pars[0]
        t_effect = np.exp(-(airt_arr-pars[1])**2/20**2/2)
        t_effect /= np.exp(-(25-pars[1])**2/20**2/2)
        par_effect = (nrad_arr/275)**max(0,pars[2])
        gppmax = t_effect*basemax*par_effect
        gpp_pred = gppmax
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),25,0.5])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
   
    basemax = pars[0]
    t_effect = np.exp(-(airt_arr-pars[1])**2/20**2/2)
    t_effect /= np.exp(-(25-pars[1])**2/20**2/2)
    par_effect = (nrad_arr/275)**max(0,pars[2])
    gppmax = t_effect*basemax*par_effect
    gpp_pred = gppmax
    dfgpp["gpp_maxpred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac
    
    dfgpp["gppR2_no_cond2"] = np.corrcoef(dfgpp.gpp_maxpred, dfgpp.gpp)[0,1]**2
    return dfgpp
#%%

def fit_gpp2(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    Ngpp_arr = np.array(dfgpp.normgpp)
    Ncond_arr = np.array(dfgpp.normcond)
    nrad_arr = np.array(dfgpp.normrad)
    
    def tofit(pars):
        basemax = pars[0]
        t_effect = np.exp(-(airt_arr-pars[2])**2/20**2/2)
        t_effect /= np.exp(-(25-pars[2])**2/20**2/2)
        slope = pars[1]*t_effect
        par_effect = (nrad_arr/275)**max(0,pars[3])
        gppmax = basemax*par_effect
        gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),120,25,0.5])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    basemax = pars[0]
    t_effect = np.exp(-(airt_arr-pars[2])**2/20**2/2)
    t_effect /= np.exp(-(25-pars[2])**2/20**2/2)
    slope = pars[1]*t_effect
    par_effect = (nrad_arr/275)**max(0,pars[3])
    gppmax = basemax*par_effect
    gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
    
    dfgpp["gppmax"] = gppmax*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
#%%
    
    def tofit(pars):
        basemax = pars[0]
        t_effect = np.exp(-(airt_arr-pars[1])**2/20**2/2)
        t_effect /= np.exp(-(25-pars[1])**2/20**2/2)
        par_effect = (nrad_arr/275)**max(0,pars[2])
        gppmax = t_effect*basemax*par_effect
        gpp_pred = gppmax
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),25,0.5])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
   
    basemax = pars[0]
    t_effect = np.exp(-(airt_arr-pars[1])**2/20**2/2)
    t_effect /= np.exp(-(25-pars[1])**2/20**2/2)
    par_effect = (nrad_arr/275)**max(0,pars[2])
    gppmax = t_effect*basemax*par_effect
    gpp_pred = gppmax
    dfgpp["gpp_maxpred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac
    
    dfgpp["gppR2_no_cond2"] = np.corrcoef(dfgpp.gpp_maxpred, dfgpp.gpp)[0,1]**2
    return dfgpp
#%%

def fit_gpp3(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    Ngpp_arr = np.array(dfgpp.normgpp)
    Ncond_arr = np.array(dfgpp.normcond)
    nrad_arr = np.array(dfgpp.normrad)
    
    def tofit(pars):
        basemax = pars[0]
        t_effect1 = np.exp(-(airt_arr-pars[2])**2/20**2/2)
        t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
        par_effect = (nrad_arr/275)**max(0,pars[3])
        gppmax = t_effect1*basemax*par_effect
        
        t_effect2 = np.exp(-(airt_arr-pars[4])**2/20**2/2)
        t_effect2 /= np.exp(-(25-pars[4])**2/20**2/2)
        slope = pars[1]*t_effect2

        gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),120,25,0.5,35])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    basemax = pars[0]
    t_effect1 = np.exp(-(airt_arr-pars[2])**2/20**2/2)
    t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
    par_effect = (nrad_arr/275)**max(0,pars[3])
    gppmax = t_effect1*basemax*par_effect
    
    t_effect2 = np.exp(-(airt_arr-pars[4])**2/20**2/2)
    t_effect2 /= np.exp(-(25-pars[4])**2/20**2/2)
    slope = pars[1]*t_effect2
    gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
    
    dfgpp["topt_max"] = pars[2]
    dfgpp["topt_slope"] = pars[4]
    dfgpp["slope"] = pars[1]
    dfgpp["par_coef"] = max(0,pars[3])
    
    dfgpp["gppmax"] = gppmax*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
#%%
    
    def tofit(pars):
        basemax = pars[0]
        t_effect = np.exp(-(airt_arr-pars[1])**2/20**2/2)
        t_effect /= np.exp(-(25-pars[1])**2/20**2/2)
        par_effect = (nrad_arr/275)**max(0,pars[2])
        gppmax = t_effect*basemax*par_effect
        gpp_pred = gppmax
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),25,0.5])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
   
    basemax = pars[0]
    t_effect = np.exp(-(airt_arr-pars[1])**2/20**2/2)
    t_effect /= np.exp(-(25-pars[1])**2/20**2/2)
    par_effect = (nrad_arr/275)**max(0,pars[2])
    gppmax = t_effect*basemax*par_effect
    gpp_pred = gppmax
    dfgpp["gpp_maxpred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac
    
    dfgpp["gppR2_no_cond2"] = np.corrcoef(dfgpp.gpp_maxpred, dfgpp.gpp)[0,1]**2
    return dfgpp


def fit_gpp_mech(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    Ngpp_arr = np.array(dfgpp.normgpp)
    Ncond_arr = np.array(dfgpp.normcond)
    nrad_arr = np.array(dfgpp.normrad)
    

    p_sfc = 101325
    o_i = 0.209*p_sfc
    tau = 2600 * 0.57**((airt_arr - 25)/10)
    Gamma_star = o_i / (2*tau)
    
    def gpp_opt(pars):
        t_effect = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
        t_effect /= np.exp(-(25-pars[1])**2/20**2/2)

        par_effect = (nrad_arr/275)**pars[2]
        ca_minus_gamma = 393 - 10*Gamma_star*pars[3]
        mm_vmax = t_effect*par_effect*ca_minus_gamma*pars[0]
        mm_k = 1.6*t_effect*par_effect
        
        gpp_pred = 0.8* mm_vmax* (1 - np.exp(-Ncond_arr/mm_k))
        return (gpp_pred-Ngpp_arr)#[np.isfinite(gpp_samp)]
    
    gpp_allmax = np.max(Ngpp_arr)
    fit0 = np.array([gpp_allmax/300,25,0.33,1])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    pars = gpp_optres.x
    
    t_effect = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
    t_effect /= np.exp(-(25-pars[1])**2/20**2/2)

    par_effect = (nrad_arr/275)**pars[2]
    ca_minus_gamma = 393 - 10*Gamma_star*pars[3]
    mm_vmax = t_effect*par_effect*ca_minus_gamma*pars[0]
    mm_k = 1.6*t_effect*par_effect
    
    gpp_pred = 0.8* mm_vmax* (1 - np.exp(-Ncond_arr/mm_k))
    
    dfgpp["gppmax"] = 0.8*mm_vmax*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["kgpp"] = mm_k*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["gpp_pred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac
    
    return dfgpp
#%%

def fit_gpp_tonly(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    Ngpp_arr = np.array(dfgpp.normgpp)
    Ncond_arr = np.array(dfgpp.normcond)
    #nrad_arr = np.array(dfgpp.normrad)
    
    nbin = 4
#    t_list = np.linspace(np.min(airt_arr),np.max(airt_arr),nbin+1)
    t_list = np.quantile(airt_arr,np.linspace(0,1,nbin+1))
    pars_list = np.zeros((nbin,2))

    for ti in range(nbin):
        tsel = (airt_arr >= t_list[ti])*(airt_arr < t_list[ti+1])
        def gpp_opt(pars):
            gpp_pred = pars[0] * (1 - np.exp(-Ncond_arr[tsel]/pars[0]*pars[1]))
            return (gpp_pred-Ngpp_arr[tsel])#[np.isfinite(gpp_samp)]
    
        gpp_allmax = np.max(Ngpp_arr)
        fit0 = np.array([gpp_allmax,110])
        gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
        pars_list[ti,:] = gpp_optres.x
        
    tcent = (t_list[1:] + t_list[:-1])/2
    intmax = np.interp(airt_arr,tcent,pars_list[:,0])
    intslope = np.interp(airt_arr,tcent,pars_list[:,1])
    gpp_pred = intmax * (1 - np.exp(-Ncond_arr/intmax*intslope))
    
    dfgpp["gppmax"] = intmax*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["kgpp"] = dfgpp["gppmax"]/intslope
    dfgpp["gpp_pred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac
    
    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
    
    return dfgpp
#%%

def fit_gpp_nopar(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    Ngpp_arr = np.array(dfgpp.normgpp)
    Ncond_arr = np.array(dfgpp.normcond)
    nrad_arr = np.array(dfgpp.normrad)

    tmean = np.mean(airt_arr)

    def tofit(pars):
        t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
        t_effect1 /= np.exp(-(tmean-pars[2])**2/20**2/2)
        
        t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
        t_effect2 /= np.exp(-(tmean-pars[3])**2/20**2/2)
        
        gppmax = pars[0] * t_effect1
        slope = pars[1] * t_effect2
        gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),120,25,35])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
    t_effect1 /= np.exp(-(tmean-pars[2])**2/20**2/2)
    
    t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
    t_effect2 /= np.exp(-(tmean-pars[3])**2/20**2/2)
    
    gppmax = pars[0] * t_effect1
    slope = pars[1] * t_effect2
    gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))

    
    dfgpp["gppmax"] = gppmax*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
    dfgpp["emax"] = pars[2]
    dfgpp["eslope"] = pars[3]
    dfgpp["slope0"] = pars[1]
    dfgpp["max0"] = pars[0]
    
    return dfgpp
#%%

def fit_gpp_noslope(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    Ngpp_arr = np.array(dfgpp.normgpp)
    Ncond_arr = np.array(dfgpp.normcond)
    nrad_arr = np.array(dfgpp.normrad)

    def tofit(pars):
        t_effect1 = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
        t_effect1 /= np.exp(-(25-pars[1])**2/20**2/2)
        
        t_effect2 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
        t_effect2 /= np.exp(-(25-pars[2])**2/20**2/2)
        
        gppmax = pars[0] * t_effect1
        slope = 110 * t_effect2
        gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))
        return gpp_pred - Ngpp_arr
    fit0 = np.array([np.max(Ngpp_arr),25,25])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    t_effect1 = np.exp(-(airt_arr-pars[1])**2 / 20**2 / 2)
    t_effect1 /= np.exp(-(25-pars[1])**2/20**2/2)
    
    t_effect2 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
    t_effect2 /= np.exp(-(25-pars[2])**2/20**2/2)
    
    gppmax = pars[0] * t_effect1
    slope = 110 * t_effect2
    gpp_pred = gppmax*(1-np.exp(-Ncond_arr/gppmax*slope))

    
    dfgpp["gppmax"] = gppmax*dfgpp.LAI*dfgpp.dayfrac
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred*dfgpp.LAI*dfgpp.dayfrac

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
    dfgpp["emax"] = pars[1]
    dfgpp["eslope"] = pars[2]
    
    return dfgpp



#%%
def fit_gpp_evi2(df1):
    dfgpp = df1.copy()
    airt_arr = np.array(dfgpp.airt)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    nrad_arr = np.array(dfgpp.normrad)
    evi_arr = np.array(dfgpp.EVI2)/np.max(dfgpp.EVI2)
    day_effect = np.array(dfgpp.dayfrac)

    def tofit(pars):
        t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
        t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
        
        t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
        t_effect2 /= np.exp(-(25-pars[3])**2/20**2/2)
        leaf_effect = 1 - (1-evi_arr)*pars[4]
        gppmax = pars[0] * t_effect1 * leaf_effect * day_effect
        slope = pars[1] * t_effect2
        gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        return gpp_pred - gpp_arr
    fit0 = np.array([np.max(gpp_arr),120,25,25,1])
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    
    t_effect1 = np.exp(-(airt_arr-pars[2])**2 / 20**2 / 2)
    t_effect1 /= np.exp(-(25-pars[2])**2/20**2/2)
    
    t_effect2 = np.exp(-(airt_arr-pars[3])**2 / 20**2 / 2)
    t_effect2 /= np.exp(-(25-pars[3])**2/20**2/2)
    
    leaf_effect = 1 - (1-evi_arr)*pars[4]
    gppmax = pars[0] * t_effect1 * leaf_effect * day_effect
    slope = pars[1] * t_effect2
    gpp_pred = gppmax*(1-np.exp(-cond_arr/gppmax*slope))
        
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = dfgpp["gppmax"]/slope
    dfgpp["gpp_pred"] = gpp_pred

    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
    dfgpp["emax"] = pars[2]
    dfgpp["eslope"] = pars[3]
    dfgpp["eamp"] = pars[4]
    
    return dfgpp
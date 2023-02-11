# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:27:50 2023

@author: natan
"""


import numpy as np
#import pymc as pm
import pandas as pd
import scipy.optimize
import statsmodels.formula.api as smf
#%%
zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf
#%%
def fit_gpp(df0,year_effect,v0,k0,fix_slope,doy_effect):
    df_to_fit = df0.copy()
    g_samp = np.array(df_to_fit.cond)
    gpp_samp = np.array(df_to_fit.gpp_qc)
    par_samp = np.array(df_to_fit.par)

    obs_gppmax = np.max(gpp_samp)
    
    doy_samp = np.array(df_to_fit.doy)
    
    #%%
    if year_effect:
    
        df_hi_cond = []
        for yearJ in pd.unique(df_to_fit.year):
            dataJ = df_to_fit.loc[df_to_fit.year == yearJ]
            year_cutoff = np.quantile(dataJ.cond,0.67)
            year_high = dataJ.loc[dataJ.cond > year_cutoff]
            df_hi_cond.append(year_high)
        df_hi_cond = pd.concat(df_hi_cond).reset_index()
#        hi_mod = smf.ols("np.log(gpp_qc) ~ np.log(par) + C(year)",data=df_hi_cond,missing="drop").fit()
        hi_mod = smf.ols("gpp_qc ~ 0 + np.sqrt(par):C(year)",data=df_hi_cond,missing="drop").fit()

    else:
        df_hi_cond = df_to_fit.loc[df_to_fit.cond > np.quantile(df_to_fit.cond,0.67)]
        #hi_mod = smf.ols("np.log(gpp_qc) ~ np.log(par)",data=df_hi_cond,missing="drop").fit()
        hi_mod = smf.ols("gpp_qc ~ 0+np.sqrt(par)",data=df_hi_cond,missing="drop").fit()

    #%%
#    gppmax = np.exp(hi_mod.predict(df_to_fit))
    gppmax0 = hi_mod.predict(df_to_fit)
    #par_exp = hi_mod.params[-1]
    #df_to_fit["par_exp"] = par_exp
    #%%
    doy_past_peak = np.array(df_to_fit.doy-df_to_fit.summer_peak)
    #%%
    # yearmat = np.array(pd.get_dummies(df_to_fit.year))

    # def gpp_opt(pars):
    #     #b_par, b_airt, b_airt2 = coefs[:3]
    #     gppmax = yearmat.dot(pars[1:])*gppmax0
    #     gpp_pred = gppmax*(1-np.exp(-g_samp*pars[0]/gppmax))
    #     return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    # fit0 = np.zeros(yearmat.shape[1]+1)
    # fit0[0] = 100
    # fit0[1:] = 1
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    # df_to_fit["gppmax"] = yearmat.dot(gpp_optres.x[1:])*gppmax0

    # kg = df_to_fit["gppmax"]/gpp_optres.x[0]
    # df_to_fit["kgpp"] =kg
    # df_to_fit["gpp_slope"] = gpp_optres.x[0]
    #%%
    if year_effect:
        gslope = 110
        yearmat = np.array(pd.get_dummies(df_to_fit.year))
    
        def gpp_opt(pars):
            #b_par, b_airt, b_airt2 = coefs[:3]
            gppmax = yearmat.dot(pars[:])*gppmax0
            gpp_pred = gppmax*(1-np.exp(-g_samp*gslope/gppmax))
            return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
        
        fit0 = np.ones(yearmat.shape[1])
        gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
        
        df_to_fit["gppmax"] = yearmat.dot(gpp_optres.x[:])*gppmax0
    
        kg = df_to_fit["gppmax"]/gslope
        df_to_fit["kgpp"] =kg
        df_to_fit["gpp_slope"] = gslope
    #%%
   

    #%%
    # gpp_pred = gppmax*(1-np.exp(-g_samp/kg))
    # df_to_fit["gpp_pred"] = gpp_pred
    # gpp_r2 = 1-np.nanmean((gpp_pred-gpp_samp)**2)/np.nanvar(gpp_samp)
    # df_to_fit["gppR2"] = gpp_r2
    #gpp_r2_null = 1-np.mean((df_to_fit.gppmax-df_to_fit.gpp)**2)/np.var(df_to_fit.gpp)
#    gpp_r2_null = np.corrcoef(gpp_samp,df_to_fit.gppmax)[0,1]**2
    #%%
    elif doy_effect:
        if fix_slope==0:
        
            def gpp_opt(pars):
                #b_par, b_airt, b_airt2 = coefs[:3]
                #year_coef = coefs[3:]
                
                gppmax2 = gppmax0*pars[1]*(1+pars[2]*doy_past_peak)
                gpp_pred = gppmax2*(1-np.exp(-g_samp*pars[0]/gppmax2))
                return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
            
            fit0 = np.ones(3)
            fit0[0] = k0
            fit0[1] = v0
            fit0[2] = -0.002
            gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm",x_scale=np.abs(fit0))#,x_scale=np.abs(fit0))
            pars = gpp_optres.x
            gppmax2 = gppmax0*pars[1]*(1+pars[2]*doy_past_peak)
            df_to_fit["gppmax"] = gppmax2
            df_to_fit["kgpp"] = gppmax2/pars[0]
            df_to_fit["gpp_slope"] = pars[0]
        else:
            def gpp_opt(pars):
                #b_par, b_airt, b_airt2 = coefs[:3]
                #year_coef = coefs[3:]
                
                gppmax2 = gppmax0*pars[0]*(1+pars[1]*doy_past_peak)
                gpp_pred = gppmax2*(1-np.exp(-g_samp*k0/gppmax2))
                return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
            
            fit0 = np.ones(2)
            fit0[0] = v0
            fit0[1] = -0.002
            gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm",x_scale=np.abs(fit0))#,x_scale=np.abs(fit0))
            pars = gpp_optres.x
            gppmax2 = gppmax0*pars[0]*(1+pars[1]*doy_past_peak)
            df_to_fit["gppmax"] = gppmax2
            df_to_fit["kgpp"] = gppmax2/k0
            df_to_fit["gpp_slope"] = k0
    
    
    else:
        
        if fix_slope:
            gslope = k0
            
            def gpp_opt(pars):
                #b_par, b_airt, b_airt2 = coefs[:3]
                #year_coef = coefs[3:]
                gpp_pred = gppmax0*pars[0]*(1-np.exp(-g_samp*gslope/gppmax0/pars[0]))
                return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
            
            fit0 = np.ones(1)
            gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
            
            df_to_fit["gppmax"] = gppmax0*gpp_optres.x[0]
        
            kg = df_to_fit["gppmax"]/gslope
            df_to_fit["kgpp"] =kg
            df_to_fit["gpp_slope"] = gslope
        
        else:
            def gpp_opt(pars):
                #b_par, b_airt, b_airt2 = coefs[:3]
                #year_coef = coefs[3:]
                gpp_pred = gppmax0*pars[0]*(1-np.exp(-g_samp*pars[1]/gppmax0/pars[0]))
                return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
            
            fit0 = np.ones(2)
            fit0[0] = v0
            fit0[1] = k0
            gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm",x_scale=np.abs(fit0))
            
            df_to_fit["gppmax"] = gppmax0*gpp_optres.x[0]
        
            kg = df_to_fit["gppmax"]/gpp_optres.x[1]
            df_to_fit["kgpp"] =kg
            df_to_fit["gpp_slope"] = gpp_optres.x[1]


    #%%
    gpp_pred = df_to_fit["gppmax"]*(1-np.exp(-g_samp/df_to_fit["kgpp"]))
    df_to_fit["gpp_pred"] = gpp_pred
    gpp_r2 = 1-np.nanmean((gpp_pred-gpp_samp)**2)/np.nanvar(gpp_samp)
    df_to_fit["gppR2"] = gpp_r2
    df_to_fit["gppR2_null"] = np.corrcoef(df_to_fit["gppmax"][np.isfinite(gpp_samp)],gpp_samp[np.isfinite(gpp_samp)])[0,1]**2

    
   #%% 
    return df_to_fit, gpp_optres
#%%


def fit_tau(dfi):


    
    wbal_samp = np.array(dfi["waterbal"])/1000
    vpd_samp = np.array(dfi.vpd)
    
    et_samp = np.array(dfi['ET'])
    k_samp = np.array(dfi["kgpp"]) #np.array(dfi["gppmax"]/gpp_optres.x[0])
    
    
    gasvol_fac_samp = np.array(dfi['gasvol_fac'])
    petVnum_samp = np.array(dfi['petVnum'])
    myga_samp = np.array(dfi['myga'])
    sV_samp = np.array(dfi['sV'])
    g_samp = np.array(dfi["cond"])
    
    #year_min = np.array(df_omit_nan.groupby("year").min()["waterbal"])
    
    fac1 = myga_samp/(22.4*gasvol_fac_samp/1000)
    g_adj = g_samp/np.sqrt(2*zsoil_mol*k_samp/(vpd_samp/100))

# yearmat = np.array(pd.get_dummies(dfi.year))
 #%%

 
 #%%
 
    def tofit(pars):
        tau = pars[0]
        slope = np.sqrt(1/(tau*(60*60*24)))
        
        smin = pars[1]
        s_adj = np.clip(wbal_samp- smin,0,width)
        final_cond = np.clip(slope*np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100)),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(2)
    fit0[0] = 30
    fit0[1] = np.quantile(wbal_samp,0.05) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    tau = pars[0]
    slope = np.sqrt(1/(tau*(60*60*24)))
    
    smin = pars[1]
    s_adj = np.clip(wbal_samp- smin,0,width)
    final_cond = np.clip(slope*np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100)),0,gmax)
    et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    
    #%%
    dfi["tau"] = tau
    #dfi["width"] = width
    dfi["smin"] = smin
    #dfi["gpp_slope"] = gpp_optres.x[0]
    #dfi["gpp_slope"] = gpp_optres.x[1]/gpp_optres.x[0]
    dfi["g_adj"] = g_adj
    dfi["et_tau"] = et_out
    dfi["etr2_smc"] = 1-np.mean((et_out-et_samp)[:]**2)/np.var(et_samp)
    #%%
    def tofit(pars):
        slope0 = pars[0]
        #gmax = pars[1]
        final_cond = np.clip(slope0*np.sqrt(k_samp*zsoil_mol/(vpd_samp/100)),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(1)
    fit0[0] = np.median(g_samp / np.sqrt(k_samp*zsoil_mol/(vpd_samp/100)))
    #%%
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    slope0 = pars[0]
    final_cond = np.clip(slope0*np.sqrt(k_samp*zsoil_mol/(vpd_samp/100)),0,gmax)
    et_null = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    dfi["etr2_null"] = 1-np.mean((et_null-et_samp)[:]**2)/np.var(et_samp)
    dfi["et_null"] = et_null
    return dfi
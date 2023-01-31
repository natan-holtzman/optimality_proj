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
import statsmodels.api as sm
import scipy.special


zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf

def fit_gpp_mm(df0,year_effect):
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
    par_exp = hi_mod.params[-1]
    df_to_fit["par_exp"] = par_exp

    #%%
    if year_effect:
        yearmat = np.array(pd.get_dummies(df_to_fit.year))
        mmfrac = 1/200
        def gpp_opt(pars):
            #b_par, b_airt, b_airt2 = coefs[:3]
            gppmax = yearmat.dot(pars[:])*gppmax0
            gpp_pred = gppmax*g_samp/(g_samp + mmfrac*gppmax)
            return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
        
        fit0 = np.ones(yearmat.shape[1])
        gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
        
        df_to_fit["gppmax"] = yearmat.dot(gpp_optres.x[:])*gppmax0
    
        kg = df_to_fit["gppmax"]*mmfrac
        df_to_fit["kgpp"] =kg
        df_to_fit["gpp_slope"] = 110
    #%%
    else:
        # def gpp_opt(pars):
        #     #b_par, b_airt, b_airt2 = coefs[:3]
        #     gppmax = pars[1]*gppmax0
        #     gpp_pred = gppmax*g_samp/(g_samp + gppmax/pars[0])
        #     return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
        
        # fit0 = np.ones(2)
        # fit0[0] = 200
        # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
        # pars = gpp_optres.x
        # df_to_fit["gppmax"] = pars[1]*gppmax0
    
        # kg = df_to_fit["gppmax"]/pars[0]
        # df_to_fit["kgpp"] =kg
    #df_to_fit["gpp_slope"] = 110
    
        def gpp_opt(pars):
            #b_par, b_airt, b_airt2 = coefs[:3]
            gppmax = pars[0]*gppmax0
            gpp_pred = gppmax*g_samp/(g_samp + gppmax/200)
            return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
        
        fit0 = np.ones(1)
        gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
        pars = gpp_optres.x
        df_to_fit["gppmax"] = pars[0]*gppmax0
    
        kg = df_to_fit["gppmax"]/200
        df_to_fit["kgpp"] =kg
#df_to_fit["gpp_slope"] = 110
    #%%
    gpp_pred = df_to_fit["gppmax"]*g_samp/(g_samp + kg)
    df_to_fit["gpp_pred"] = gpp_pred
    gpp_r2 = 1-np.nanmean((gpp_pred-gpp_samp)**2)/np.nanvar(gpp_samp)
    df_to_fit["gppR2"] = gpp_r2
    df_to_fit["gppR2_null"] = np.corrcoef(gppmax0[np.isfinite(gpp_samp)],gpp_samp[np.isfinite(gpp_samp)])[0,1]**2
    
   #%% 
    return df_to_fit
#%%


def fit_tau_mm(dfi):


    
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
    g_n = dfi.cond/dfi.kgpp
    gn2 = g_n - np.log(g_n+1)
    lhs_gg = gn2*2*k_samp*(vpd_samp/100)/zsoil_mol
    invmod_mm = sm.OLS(lhs_gg,sm.add_constant(wbal_samp)).fit()

    tau_guess = 1/invmod_mm.params[1]/(60*60*24)
    smin_guess = -invmod_mm.params[0]/invmod_mm.params[1]
    #dfi["gslope_mm"] = dfgpp_mm.gpp_slope.iloc[0]
    #dfi["gppmax_mm"] = dfgpp_mm.gppmax
 
 #%%
 
    def tofit(pars):
        tau_s = pars[0]*(60*60*24)
        #slope = np.sqrt(1/(tau*(60*60*24)))
        
        smin = pars[1]
        s_adj = np.clip(wbal_samp- smin,0,width)
        inside_term = zsoil_mol*s_adj/(vpd_samp/100)/(2*k_samp*tau_s)
        
        inside_w = np.exp(-1*inside_term-1)
        final_cond = -k_samp *(1+np.real(scipy.special.lambertw(-inside_w,-1)))
        final_cond[np.isnan(final_cond)] = 0
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(2)
    fit0[0] = tau_guess
    fit0[1] = smin_guess #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    tau, smin = cond_optres.x
    tau_s = tau*(60*60*24)
    #slope = np.sqrt(1/(tau*(60*60*24)))
    
    s_adj = np.clip(wbal_samp- smin,0,width)
    inside_term = zsoil_mol*s_adj/(vpd_samp/100)/(2*k_samp*tau_s)
    
    inside_w = np.exp(-1*inside_term-1)
    final_cond = -k_samp *(1+np.real(scipy.special.lambertw(-inside_w,-1)))
    final_cond[np.isnan(final_cond)] = 0

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
    s_const = np.mean(s_adj)
    def tofit(pars):
        tau_s = pars[0]*(60*60*24)
        #slope = np.sqrt(1/(tau*(60*60*24)))
        
        inside_term = zsoil_mol*s_const/(vpd_samp/100)/(2*k_samp*tau_s)
        
        inside_w = np.exp(-1*inside_term-1)
        final_cond = -k_samp *(1+np.real(scipy.special.lambertw(-inside_w,-1)))
        final_cond[np.isnan(final_cond)] = 0
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(1)
    fit0[0] = tau*1
    #%%
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    tau_s = pars[0]*(60*60*24)
    #slope = np.sqrt(1/(tau*(60*60*24)))
    
    inside_term = zsoil_mol*s_const/(vpd_samp/100)/(2*k_samp*tau_s)
    
    inside_w = np.exp(-1*inside_term-1)
    final_cond = -k_samp *(1+np.real(scipy.special.lambertw(-inside_w,-1)))
    final_cond[np.isnan(final_cond)] = 0
    et_null = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    dfi["et_null"] = et_null
    dfi["etr2_null"] = 1-np.mean((et_null-et_samp)[:]**2)/np.var(et_samp)

    return dfi
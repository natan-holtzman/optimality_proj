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

#%%
zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
#gmax = np.inf
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
 
    tau_hi = 1/(invmod_mm.params[1]-2*invmod_mm.bse[1])/(60*60*24)
    tau_lo = 1/(invmod_mm.params[1]+2*invmod_mm.bse[1])/(60*60*24)
 
 #%%
 
    def tofit(pars):
        tau_s = pars[0]*(60*60*24)
        #slope = np.sqrt(1/(tau*(60*60*24)))
        
        smin = pars[1]
        gmax = pars[2]
        gmin = pars[3]
        s_adj = np.clip(wbal_samp- smin,0,width)
        inside_term = zsoil_mol*s_adj/(vpd_samp/100)/(2*k_samp*tau_s)
        
        inside_w = np.exp(-1*inside_term-1)
        final_cond = -k_samp *(1+np.real(scipy.special.lambertw(-inside_w,-1)))
        final_cond[np.isnan(final_cond)] = 0
        final_cond = np.clip(final_cond,gmin,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(4)
    fit0[0] = tau_guess
    fit0[1] = smin_guess #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    fit0[2] = np.quantile(g_samp,0.95)
    fit0[3] = np.quantile(g_samp,0.05)
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    tau, smin, gmax, gmin = cond_optres.x
    tau_s = tau*(60*60*24)
    #slope = np.sqrt(1/(tau*(60*60*24)))
    
    s_adj = np.clip(wbal_samp- smin,0,width)
    inside_term = zsoil_mol*s_adj/(vpd_samp/100)/(2*k_samp*tau_s)
    
    inside_w = np.exp(-1*inside_term-1)
    final_cond = -k_samp *(1+np.real(scipy.special.lambertw(-inside_w,-1)))
    final_cond[np.isnan(final_cond)] = 0
    final_cond = np.clip(final_cond,gmin,gmax)

    et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
 
    #%%
    dfi["tau"] = tau
    dfi["tau_reg"] = tau_guess
    #dfi["tau_year"] = tau_guess_year
    dfi["tau_hi"] = tau_hi
    dfi["tau_lo"] = tau_lo
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
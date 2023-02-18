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
#gmax = np.inf
#%%

def fit_tau_res(dfi):
#%%
    rescond = dfi.res_cond.iloc[0]
    
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
        beta_s = 1/(tau*(60*60*24))
        
        smin = pars[1]
        gmax = pars[2]
        s_adj = np.clip(wbal_samp- smin,0,width)
        final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(3)
    fit0[0] = 50
    #fit0[1] = np.quantile(wbal_samp,0.25) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    fit0[1] = np.min(wbal_samp)#-0.1 #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)

    fit0[2] = 0.4 #np.quantile(g_samp,0.95) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    tau = pars[0]
    beta_s = 1/(tau*(60*60*24))
    
    smin = pars[1]
    gmax = pars[2]
    s_adj = np.clip(wbal_samp- smin,0,width)
    final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,gmax)
    et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    
    dfi["pred_cond"] = final_cond
    #%%
    dfi["gmax"] = gmax
    #dfi["gmin"] = gmin

    dfi["tau"] = tau
    #dfi["width"] = width
    dfi["smin"] = smin
    #dfi["gpp_slope"] = gpp_optres.x[0]
    #dfi["gpp_slope"] = gpp_optres.x[1]/gpp_optres.x[0]
    dfi["g_adj"] = g_adj
    dfi["et_tau"] = et_out
    dfi["etr2_smc"] = 1-np.mean((et_out-et_samp)[:]**2)/np.var(et_samp)
    #%%
    s_adj_const = np.mean(s_adj)
    def tofit(pars):
        tau0 = pars[0]
        beta_s = 1/(tau0*(60*60*24))
        
        gmax = pars[1]
        final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj_const/(vpd_samp/100) + rescond**2),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(2)
    fit0[0] = 1*tau
    fit0[1] = 0.4 #np.quantile(g_samp,0.95)

    #%%
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    #%%
    tau0 = pars[0]
    beta_s = 1/(tau0*(60*60*24))
    
    gmax = pars[1]
    final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj_const/(vpd_samp/100) + rescond**2),0,gmax)
    et_null = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    dfi["etr2_null"] = 1-np.mean((et_null-et_samp)[:]**2)/np.var(et_samp)    
    dfi["et_null"] = et_null
    return dfi
#%%

def fit_tau_res_width(dfi):
#%%
    rescond = dfi.res_cond.iloc[0]
    
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
        beta_s = 1/(tau*(60*60*24))
        
        smin = pars[1]
        w0 = pars[2]
        s_adj = np.clip(wbal_samp- smin,0,w0)
        final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,np.inf)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(3)
    fit0[0] = 50
    #fit0[1] = np.quantile(wbal_samp,0.25) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    fit0[1] = np.min(wbal_samp)#-0.1 #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)

    fit0[2] = 0.1 #np.quantile(g_samp,0.95) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    tau = pars[0]
    beta_s = 1/(tau*(60*60*24))
    
    smin = pars[1]
    w0 = pars[2]
    s_adj = np.clip(wbal_samp- smin,0,w0)
    final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,np.inf)
    et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    
    #dfi["pred_cond"] = np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100))
    #%%
    #dfi["gmax"] = gmax
    #dfi["gmin"] = gmin

    dfi["tau"] = tau
    dfi["width"] = w0
    dfi["smin"] = smin
    #dfi["gpp_slope"] = gpp_optres.x[0]
    #dfi["gpp_slope"] = gpp_optres.x[1]/gpp_optres.x[0]
    dfi["g_adj"] = g_adj
    dfi["et_tau"] = et_out
    dfi["etr2_smc"] = 1-np.mean((et_out-et_samp)[:]**2)/np.var(et_samp)
    #%%
    s_adj_const = np.mean(s_adj)
    def tofit(pars):
        tau0 = pars[0]
        beta_s = 1/(tau0*(60*60*24))
        
        final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj_const/(vpd_samp/100) + rescond**2),0,np.inf)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(1)
    fit0[0] = 1*tau

    #%%
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    #%%
    tau0 = pars[0]
    beta_s = 1/(tau0*(60*60*24))
    
    final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj_const/(vpd_samp/100) + rescond**2),0,np.inf)
    et_null = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    dfi["etr2_null"] = 1-np.mean((et_null-et_samp)[:]**2)/np.var(et_samp)    
    dfi["et_null"] = et_null
    return dfi
#%%


#%%
def fit_assume_tau_res(dfi,tau):
#%%
    rescond = dfi.res_cond.iloc[0]
    
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
    beta_s = 1/(tau*(60*60*24))
    def tofit(pars):        
        smin = pars[0]
        gmax = pars[1]
        s_adj = np.clip(wbal_samp- smin,0,width)
        final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(2)
    #fit0[1] = np.quantile(wbal_samp,0.25) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    fit0[0] = np.min(wbal_samp)#-0.1 #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    fit0[1] = 0.4 #np.quantile(g_samp,0.95) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    smin = pars[0]
    gmax = pars[1]
    s_adj = np.clip(wbal_samp- smin,0,width)
    final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,gmax)
    et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    
    #dfi["pred_cond"] = np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100))
    #%%
    dfi["gmax2"] = gmax
    #dfi["gmin"] = gmin

    dfi["tau2"] = tau
    #dfi["width"] = width
    dfi["smin2"] = smin
    #dfi["gpp_slope"] = gpp_optres.x[0]
    #dfi["gpp_slope"] = gpp_optres.x[1]/gpp_optres.x[0]
    dfi["et_tau2"] = et_out
    dfi["etr2_smc2"] = 1-np.mean((et_out-et_samp)[:]**2)/np.var(et_samp)
    #%%
    
    return dfi
#%%

def fit_tau_res_assume_max(dfi,gmax):
#%%
    rescond = dfi.res_cond.iloc[0]
    
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
        beta_s = 1/(tau*(60*60*24))
        
        smin = pars[1]
        s_adj = np.clip(wbal_samp- smin,0,width)
        final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(2)
    fit0[0] = 50
    #fit0[1] = np.quantile(wbal_samp,0.25) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    fit0[1] = np.min(wbal_samp)#-0.1 #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    tau = pars[0]
    beta_s = 1/(tau*(60*60*24))
    
    smin = pars[1]
    s_adj = np.clip(wbal_samp- smin,0,width)
    final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj/(vpd_samp/100) + rescond**2),0,gmax)
    et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    
    #dfi["pred_cond"] = np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100))
    #%%
    dfi["gmax"] = gmax
    #dfi["gmin"] = gmin

    dfi["tau"] = tau
    #dfi["width"] = width
    dfi["smin"] = smin
    #dfi["gpp_slope"] = gpp_optres.x[0]
    #dfi["gpp_slope"] = gpp_optres.x[1]/gpp_optres.x[0]
    dfi["g_adj"] = g_adj
    dfi["et_tau"] = et_out
    dfi["etr2_smc"] = 1-np.mean((et_out-et_samp)[:]**2)/np.var(et_samp)
    #%%
    s_adj_const = np.mean(s_adj)
    def tofit(pars):
        tau0 = pars[0]
        beta_s = 1/(tau0*(60*60*24))
        
        final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj_const/(vpd_samp/100) + rescond**2),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
        
    fit0 = np.zeros(1)
    fit0[0] = 1*tau

    #%%
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    pars = cond_optres.x
    #%%
    tau0 = pars[0]
    beta_s = 1/(tau0*(60*60*24))
    
    final_cond = np.clip(np.sqrt(2*beta_s*k_samp*zsoil_mol*s_adj_const/(vpd_samp/100) + rescond**2),0,gmax)
    et_null = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    dfi["etr2_null"] = 1-np.mean((et_null-et_samp)[:]**2)/np.var(et_samp)    
    dfi["et_null"] = et_null
    return dfi

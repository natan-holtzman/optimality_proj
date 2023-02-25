# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:04:24 2023

@author: natan
"""


import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import statsmodels.api as sm
import scipy.optimize
import glob
import statsmodels.formula.api as smf

import matplotlib as mpl
import h5py
#%%
from fit_tau_res_cond2 import fit_tau_res , fit_tau_res_assume_max, fit_tau_res_assume_max_smin, fit_tau_res_width

#%%
do_bif = 0
if do_bif:
    biftab = pd.read_excel(r"C:\Users\nholtzma\Downloads\fluxnet2015\FLX_AA-Flx_BIF_ALL_20200501\FLX_AA-Flx_BIF_DD_20200501.xlsx")
    groups_to_keep = ["GRP_CLIM_AVG","GRP_HEADER","GRP_IGBP","GRP_LOCATION","GRP_SITE_CHAR"]#,"GRP_LAI","GRP_ROOT_DEPTH","SOIL_TEX","SOIL_DEPTH"]
    biftab = biftab.loc[biftab.VARIABLE_GROUP.isin(groups_to_keep)]
    bif2 = biftab.pivot_table(index='SITE_ID',columns="VARIABLE",values="DATAVALUE",aggfunc="first")
    bif2.to_csv("fn2015_bif_tab.csv")
#%%
def cor_skipna(x,y):
    goodxy = np.isfinite(x*y)
    return scipy.stats.pearsonr(x[goodxy],y[goodxy])
#%%
bif_data = pd.read_csv("fn2015_bif_tab_h.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])].copy()
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])].copy()
metadata = pd.read_csv("fluxnet_site_info_all.csv")
#%%

#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%
#width = 1
zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
#gmax = np.inf
#%%
rain_dict = {}
year_tau_dict = {}
site_result = {}
#%%

sites_late_season = pd.read_csv("gs_50_50_include_unbalance9.csv")
sites_late_season["res_cond"] = 0
#%%
sites_late_season = sites_late_season.loc[sites_late_season["gpp"]  > 0]
sites_late_season = pd.merge(sites_late_season,bif_forest,on="SITE_ID",how='left')
#%%
sites_late_season["drel_spring"] = -np.clip(sites_late_season["doy"] - sites_late_season["summer_peak"],-np.inf,0) / (sites_late_season["summer_peak"] - sites_late_season["summer_start"])
sites_late_season["drel_fall"] = np.clip(sites_late_season["doy"] - sites_late_season["summer_peak"],0,np.inf) / (sites_late_season["summer_end"] - sites_late_season["summer_peak"])
sites_late_season["drel_both"] = -sites_late_season["drel_spring"] + sites_late_season["drel_fall"]
#%%



#%%
all_results = []

for site_id in pd.unique(sites_late_season.SITE_ID)[:]:#[forest_daily[x] for x in [70,76]]:
    print(site_id)
    #%%
    dfgpp = sites_late_season.loc[sites_late_season.SITE_ID==site_id].reset_index()
    dfgpp = dfgpp.loc[dfgpp.cond < np.quantile(dfgpp.cond,0.95)].reset_index()
    
    
    gpparr = np.array(dfgpp.gpp)
    condarr = np.array(dfgpp.cond)
    tarr = np.array(dfgpp.airt)
    par_arr = np.array(dfgpp.par)
    drel_arr = np.array(dfgpp.drel_both)
    
    
    def tofit(pars):
        gppmax = pars[0] * (par_arr/550)**pars[1] * (1-pars[2]*drel_arr**2)
        slope = 100 * np.exp(-((tarr-pars[3])/pars[4]/2)**2)
        gpp_pred = gppmax * (1 - np.exp(-condarr/gppmax*slope))
        return gpp_pred - gpparr
    
    fit0 = np.array([np.quantile(dfgpp.gpp,0.9),0.5,0.1,25,15])
    optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = optres.x
    gppmax = pars[0] * (dfgpp.par/550)**pars[1] * (1-pars[2]*dfgpp.drel_both**2)
    slope =  100 * np.exp(-((dfgpp.airt-pars[3])/pars[4]/2)**2)
    gpp_pred = gppmax * (1 - np.exp(-dfgpp.cond/gppmax*slope))
    #%%
    dfgpp["gpp_pred_0"] = gpp_pred*1
    dfgpp["slope_0"] = slope*1
    dfgpp["gppmax_0"] = gppmax*1
    
    dfgpp["gppmax_base"] = pars[0]
    dfgpp["par_power_550"] = pars[1]
    dfgpp["season_coef"] = pars[2]
    dfgpp["airt_center"] = pars[3]
    dfgpp["airt_bw"] = pars[4]
    
    
    #%%
    dfgpp["log_resid"] = np.log(dfgpp.gpp/gpp_pred)
    dfyear = dfgpp.groupby("year").median(numeric_only=True).reset_index()
    #dfgpp = pd.merge(dfgpp,dfyear[["year","log_resid"]],how='left',on='year')
    #%%
    year_mat = np.array(pd.get_dummies(dfgpp.year))
    year_ratio = np.exp(np.array(dfyear.log_resid))

    #%%
    parsY = np.ones(year_mat.shape[1])
    for yi in range(year_mat.shape[1]):
        ysel = year_mat[:,yi]==1
        gppmax_Y = gppmax[ysel]
        gppobs_Y = gpparr[ysel]
        cond_Y = condarr[ysel]
        slope_Y = slope[ysel]
        def tofit(pars):
            gpp_pred = pars[0]*gppmax_Y * (1 - np.exp(-cond_Y/gppmax_Y/pars[0]*slope_Y))
            return gpp_pred - gppobs_Y
        fit0 = np.array([year_ratio[yi]])
        optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
        parsY[yi] = optres.x
        #%%
    gppmax2 = gppmax*year_mat.dot(parsY)
    gpp_pred2 = gppmax2 * (1 - np.exp(-dfgpp.cond/gppmax2*slope))
    #%%
    dfgpp["gpp_pred"] = gpp_pred2
    dfgpp["slope"] = slope
    dfgpp["gppmax"] = gppmax2
    #%%
    all_results.append(dfgpp)
all_results = pd.concat(all_results)
#%%
all_results.to_csv("sitewise_gpp_with_year.csv")
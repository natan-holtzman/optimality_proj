# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:46:32 2022

@author: nholtzma
"""

#what if we have one set of params for forest/woody savanna
#and one for other biomes?
#or just for slope?


#fname = r"C:\Users\nholtzma\Downloads\fluxnet2015\FLX_US-Me2_FLUXNET2015_SUBSET_2002-2014_1-4\FLX_US-Me2_FLUXNET2015_SUBSET_DD_2002-2014_1-4.csv"
#fname = r"C:\Users\natan\OneDrive - Stanford\Documents\moflux_docs\mdp_experiment\AMF_US-Me2_FLUXNET_SUBSET_DD_2002-2020_3-5.csv"
#fname = r"C:\Users\natan\OneDrive - Stanford\Documents\moflux_docs\mdp_experiment\AMF_US-MOz_FLUXNET_SUBSET_DD_2004-2019_3-5.csv"
#fname = r"C:\Users\natan\OneDrive - Stanford\Documents\moflux_docs\mdp_experiment\AMF_US-MMS_FLUXNET_SUBSET_DD_1999-2020_3-5.csv"

import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import statsmodels.api as sm
import scipy.optimize
import glob
import statsmodels.formula.api as smf

import matplotlib as mpl
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
bif_data = pd.read_csv("fn2015_bif_tab.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])]
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])]
metadata = pd.read_csv("fluxnet_site_info_all.csv")
#bif_forest = bif_forest.loc[~bif_forest.SITE_ID.isin(["IT-CA1","IT-CA3","AU-Emr"])]
#all_daily = glob.glob("daily_data\*.csv")
#forest_daily = [x for x in all_daily if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
#%%

evidata = pd.read_csv("meanEVI_pheno.csv")
evidata["EVImax"] = (evidata["EVI_Minimum_1"] + evidata["EVI_Amplitude_1"])/10000
evidata["EVIamp"] = (evidata["EVI_Amplitude_1"])/10000

#evidata["EVImax2"] = (evidata["EVI_Minimum_2"] + evidata["EVI_Amplitude_2"])/10000
#evi_tab = dict(zip(evidata.SITE_ID,evidata.EVImax))
#evi_amp_tab = dict(zip(evidata.SITE_ID,evidata["EVI_Amplitude_1"]/10000))

#year_results = pd.merge(year_results,evidata,on="SITE_ID",how="left")

#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%
#filename_list = glob.glob("processed_nov7b/*.csv")
#from functions_from_nov16 import fit_gpp, fit_tau

# from double_doy_effect_gmax_gmin import fit_gpp, fit_tau
# from fit_gpp_use_evi import fit_gpp_evi
# from fit_gpp_airt_slope import fit_gpp_setslope
# from simple_gpp_mod import fit_gpp_simple

# from gmax_gmin_doy_mm import fit_gpp_mmY

# from use_w_function import fit_gpp_mm, fit_tau_mm
# from use_w_function_reg import fit_tau_mm_reg
#%%
#%%
#sites_late_season = pd.read_csv("late_summer_data.csv")
#rain_data = pd.read_csv("rain_late_season.csv")
#%%
#sites_late_season = pd.read_csv("gs_50_50_daynight_data.csv")
dfgpp_all = pd.read_csv("gs_50_50_sinterp_summer.csv")
dfgpp_all = pd.merge(dfgpp_all,bif_data,on="SITE_ID",how="left")

simple_biomes = {"SAV":"Savanna",
                 "WSA":"Savanna",
                 "CSH":"Shrubland",
                 "OSH":"Shrubland",
              "EBF":"Evergreen broadleaf forest",
              "ENF":"Evergreen needleleaf forest",
              "GRA":"Grassland",
              "DBF":"Deciduous broadleaf forest",
              "MF":"Mixed forest",
              }
biome_list = ["Evergreen needleleaf forest", "Mixed forest", "Deciduous broadleaf forest", "Evergreen broadleaf forest",
              "Grassland","Shrubland","Savanna"]
#%%
dfgpp_all["combined_biome"] = [simple_biomes[x] for x in dfgpp_all["IGBP"]]

allmax = dfgpp_all.groupby("SITE_ID").max(numeric_only="True").reset_index().rename(columns={"gpp":"gpp_max_obs"})
dfgpp_all = pd.merge(dfgpp_all,allmax[["SITE_ID","gpp_max_obs"]],how="left",on="SITE_ID")

dfgpp_all = dfgpp_all.loc[dfgpp_all.gpp_max_obs > 5].reset_index()

#dfgpp["is_woody"] = dfgpp.IGBP.isin(["MF","ENF","EBF","DBF","DNF","WSA"])

#dfgpp = dfgpp.loc[~dfgpp.is_woody]
all_biomes = []
for x in biome_list:
    dfgpp = dfgpp_all.loc[dfgpp_all.combined_biome==x].copy()

#rain_data = pd.read_csv("rain_50_50_data.csv")
#%%

    dfgpp.gpp = (2*dfgpp.gpp_qc + 0*dfgpp.gpp_nt)/2
    dfgpp = dfgpp.loc[np.isfinite(dfgpp.gpp_qc)].copy()
    #%%
    # dfnew = []
    # for site_id in pd.unique(dfgpp.SITE_ID):
    #     #site_id = "US-MMS"
    #     print(site_id)
    #     site_subset = dfgpp.loc[dfgpp.SITE_ID==site_id].copy()
    #     site_subset["cond_gtm"] = np.clip(site_subset.cond - np.median(site_subset.cond),0,np.inf)
    #     mp = smf.ols("gpp ~ cond + cond_gtm",data=site_subset).fit()
    #%%
    dfgpp["doy_late_rel"] = np.clip(np.array(dfgpp.doy-dfgpp.summer_peak),0,300)/np.array(dfgpp.summer_end-dfgpp.summer_peak)
    dfgpp["doy_early_rel"] = -np.clip(np.array(dfgpp.doy-dfgpp.summer_peak),-300,0)/np.array(dfgpp.summer_peak-dfgpp.summer_start)
    dfgpp["doy_beginning"] = np.clip(dfgpp["doy_early_rel"]-0.75, 0,0.25)
    #%%
    #dfgpp = dfgpp.loc[dfgpp["EVImax"] > 0.3]
    #dfgpp["gpp_dpar"] = dfgpp.gpp/dfgpp.par
    #dfgpp["gpp_dspar"] = dfgpp.gpp/np.sqrt(dfgpp.par)
    
    #%%
    #dfgpp["evi_lt3"] = np.clip(dfgpp.EVImax,0,0.3)
    #dfgpp["evi_gt5"] = np.clip(dfgpp.EVImax,0.5,1)
    #site_year_max = dfgpp.groupby(["SITE_ID","year"]).max(numeric_only=True).reset_index()
    #site_year_min = dfgpp.groupby(["SITE_ID","year"]).min(numeric_only=True).reset_index()
    
    site_year_max = dfgpp.groupby(["SITE_ID","year"]).quantile(0.9,numeric_only=True).reset_index()
    site_year_min = dfgpp.groupby(["SITE_ID","year"]).quantile(0.1,numeric_only=True).reset_index()
    
    
    
    site_year_max["site_amp"] = site_year_max.gpp - site_year_min.gpp
    
    sym_mean = site_year_max.groupby("SITE_ID").mean(numeric_only=True).reset_index()
    #%%
    site_matrix = np.array(pd.get_dummies(dfgpp.SITE_ID))
    #%%
    
    site_coef = np.array(sym_mean.gpp)
    #site_coef = 0*site_coef + np.mean(site_coef)
    site_effect = site_matrix.dot(site_coef)
    
    #%%
    #dfgpp["g_unc"] = np.clip(dfgpp.et_unc/dfgpp.vpd*100,0.0025,np.inf)
    dfgpp["g_unc"] = np.clip(dfgpp.gpp_unc*dfgpp.gpp,0.05,np.inf)
    #dfgpp["g_unc"] = np.clip(dfgpp.nee_unc,0.1,np.inf)
    #dfgpp["g_unc"] = np.clip(dfgpp["g0_unc"]*dfgpp["a_unc"],np.exp(-7),np.inf)
    #%%
    
    #%%
    erel = np.array(dfgpp["doy_early_rel"])
    lrel = np.array(dfgpp["doy_late_rel"])
    brel = np.array(dfgpp["doy_beginning"])
    
    airt_arr = np.array(dfgpp["airt"])
    par_arr = np.array(dfgpp['par'])
    cond_arr = np.array(dfgpp["cond"])
    gpp_arr = np.array(dfgpp["gpp"])
    #%%
    def tofit(pars):
        par_exp = pars[0]
        airt_center = pars[1]
        airt_bw = pars[2]
        airt_effect = np.exp(-(airt_arr-airt_center)**2 / (2*airt_bw**2))
        slope_val = pars[3]
        
        amp_spring = pars[4]
        amp_fall = pars[5]
        amp_early = pars[6]
        season_effect = (1 - amp_spring*erel - amp_early*brel) * (1 - amp_fall*lrel)
        mpar = pars[7]
        
        
        #site_coef = pars[6:]
        #site_effect = site_matrix.dot(site_coef)
        #amp_spring = np.ones(site_matrix.shape[1])*0.5
        #amp_fall = np.ones(site_matrix.shape[1])*0.5
        #season_template = site_matrix.dot(site_coef) * (1 - site_matrix.dot(amp_spring)*dfgpp["doy_early_rel"]) * (1 - site_matrix.dot(amp_fall)*dfgpp["doy_late_rel"])
        
        
    
        Amax = mpar*site_effect*season_effect*par_arr/(par_arr+par_exp)#*airt_effect
        K = Amax/(slope_val*airt_effect)
    
        gpp_pred = Amax*(1-np.exp(-cond_arr/K))
        resid = (gpp_pred-gpp_arr)/dfgpp.g_unc
        return resid#[np.isfinite(dfgpp["gpp"])]
    #%%
    fit0 = np.array([300,20,15,100,0.2,0.2,0.2,1])
    #fit0[6:] = np.array(sym_mean.gpp)
    #%%
    gpp_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    #%%
    pars = gpp_optres.x
    #%%
    par_exp = pars[0]
    airt_center = pars[1]
    airt_bw = pars[2]
    airt_effect = np.exp(-(airt_arr-airt_center)**2 / (2*airt_bw**2))
    slope_val = pars[3]
    
    amp_spring = pars[4]
    amp_fall = pars[5]
    amp_early = pars[6]
    season_effect = (1 - amp_spring*erel - amp_early*brel) * (1 - amp_fall*lrel)
    mpar = pars[7]
    
    
    #site_coef = pars[6:]
    #site_effect = site_matrix.dot(site_coef)
    #amp_spring = np.ones(site_matrix.shape[1])*0.5
    #amp_fall = np.ones(site_matrix.shape[1])*0.5
    #season_template = site_matrix.dot(site_coef) * (1 - site_matrix.dot(amp_spring)*dfgpp["doy_early_rel"]) * (1 - site_matrix.dot(amp_fall)*dfgpp["doy_late_rel"])
    
    Amax = mpar*site_effect*season_effect*par_arr/(par_arr+par_exp)#*airt_effect
    K = Amax/(slope_val*airt_effect)
    
    gpp_pred = Amax*(1-np.exp(-cond_arr/K))
    #%%
    dfgpp["gppmax"] = Amax
    dfgpp["gpp_slope"] = slope_val*airt_effect
    dfgpp["gpp_pred"] = gpp_pred
    dfgpp["kgpp"] = dfgpp["gppmax"] / dfgpp["gpp_slope"]
    #%%
    dfgpp["season_effect"] = season_effect
    #dfgpp["evi_template"] = evi_template
    #%%
    
    #%%
    dfnew = []
    for site_id in sym_mean.SITE_ID:
        #site_id = "US-MMS"
        print(site_id)
        site_subset = dfgpp.loc[dfgpp.SITE_ID==site_id].copy()
        #max_no_season = site_subset.gppmax/site_subset.season_effect
        #max_no_par = site_subset.gppmax
        def tofit_site(pars):
            #par_effect
            newmax = pars[0]*site_subset.gppmax
            #newslope = site_subset.gpp_slope#*pars[3]
            gpp_pred_site = newmax*(1-np.exp(-site_subset.cond/newmax*site_subset.gpp_slope))
            return (gpp_pred_site-site_subset.gpp)/site_subset.g_unc
        fit0 = np.array([1])
        gpp_optres = scipy.optimize.least_squares(tofit_site,x0=fit0,method="lm",x_scale=np.abs(fit0))
        
        pars = gpp_optres.x
        newmax = pars[0]*site_subset.gppmax
        gpp_pred_site = newmax*(1-np.exp(-site_subset.cond/newmax*site_subset.gpp_slope))
        #site_subset["season_effect"] = site_season_eff
        site_subset["gppmax"] = newmax
        site_subset["gpp_pred"] = gpp_pred_site
        site_subset["gpp_slope_base"] = slope_val#*pars[3]
        #site_subset["gpp_slope_with_airt"] = newslope
        site_subset["kgpp"] = newmax/site_subset.gpp_slope
        dfnew.append(site_subset)
    #%%
    
        #%%
    newgpp = pd.concat(dfnew)
    #%%
    #newgpp = newgpp.loc[newgpp.gpp_slope_base < 500].copy()
    #%%
    newgpp["par_effect"] = newgpp.par/(newgpp.par+par_exp)
    newgpp["gpp_max_base"] = newgpp.gppmax/newgpp.season_effect/newgpp.par_effect
    #%%
    all_biomes.append(newgpp)
#%%
dfgpp = pd.concat(all_biomes)
#%%
plt.figure()
plt.plot(dfgpp.gpp_pred,dfgpp.gpp,'.',alpha=0.1)
plt.plot([0,15],[0,15])
#%%
dfgpp = dfgpp.loc[dfgpp.gpp_slope_base < 1000]
#%%
#df_allmean = newgpp.groupby(["SITE_ID"]).mean(numeric_only=True).reset_index()
#df_allmean = pd.merge(df_allmean,bif_data,on="SITE_ID",how="left")

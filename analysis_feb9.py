# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:46:32 2022

@author: nholtzma
"""




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
bif_data = pd.read_csv("fn2015_bif_tab_h.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])]
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])]
metadata = pd.read_csv("fluxnet_site_info_all.csv")
#bif_forest = bif_forest.loc[~bif_forest.SITE_ID.isin(["IT-CA1","IT-CA3","AU-Emr"])]
#all_daily = glob.glob("daily_data\*.csv")
#forest_daily = [x for x in all_daily if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
#%%

# evidata = pd.read_csv("meanEVI_pheno.csv")
# evidata["EVImax"] = (evidata["EVI_Minimum_1"] + evidata["EVI_Amplitude_1"])/10000
# #evidata["EVImax2"] = (evidata["EVI_Minimum_2"] + evidata["EVI_Amplitude_2"])/10000
# evi_tab = dict(zip(evidata.SITE_ID,evidata.EVImax))
# evi_amp_tab = dict(zip(evidata.SITE_ID,evidata["EVI_Amplitude_1"]/10000))

# #year_results = pd.merge(year_results,evidata,on="SITE_ID",how="left")

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

from double_doy_effect_gmax_gmin2 import fit_tau, fit_tau_width
# from fit_gpp_use_evi import fit_gpp_evi
# from fit_gpp_airt_slope import fit_gpp_setslope
# from simple_gpp_mod import fit_gpp_simple

# from gmax_gmin_doy_mm import fit_gpp_mmY

from mm_gmax_gmin import fit_tau_mm
# from use_w_function_reg import fit_tau_mm_reg

#%%
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
all_results = []
site_result = {}
#%%
#sites_late_season = pd.read_csv("late_summer_data.csv")
#rain_data = pd.read_csv("rain_late_season.csv")
#%%
#sites_late_season = pd.read_csv("gs_50_50_daynight_data.csv")
#sites_late_season = pd.read_csv("data_gpp_sitewise_nosub_mm2.csv")
sites_late_season = pd.read_csv("data_gpp_weight_slope.csv")
#sites_late_season = pd.read_csv("data_gpp_evi_parsq.csv")

#sites_late_season["kgpp"] = 1*sites_late_season["kgpp2"]
#sites_late_season["gpp_pred"] = 1*sites_late_season["gpp_pred2"]

#sites_late_season["kgpp"] = sites_late_season["kgpp"]/sites_late_season["season_effect"]


rain_data = pd.read_csv("rain_50_50_nosub.csv")
#%%
#sites_late_season = sites_late_season.loc[np.isfinite(sites_late_season.EVImax)]
#sites_late_season = sites_late_season.loc[sites_late_season.EVImax > 0.3]


#%%
for site_id in pd.unique(sites_late_season.SITE_ID)[:]:#[forest_daily[x] for x in [70,76]]:
#%%
    print(site_id)
    dfgpp = sites_late_season.loc[sites_late_season.SITE_ID==site_id].copy()
    dfgpp = dfgpp.loc[dfgpp.kgpp > 0].copy()
    #nyear_in_data = len(pd.unique(df_to_fit.year))
    #%%
    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2


    #rescond = cor_skipna(dfgpp.cond,  dfgpp.gpp_pred-dfgpp.gpp)
    #resdoy = cor_skipna(dfgpp.doy,  dfgpp.gpp_pred-dfgpp.gpp)
    
    #dfgpp["cor_gres_cond"] = rescond[0]
    #dfgpp["cor_gres_doy"] = resdoy[0]
    
    #if rescond.pvalue < 0.05 or resdoy.pvalue < 0.05:
    #    print("Unexplained GPP variation")
    #    continue
    #df_to_fit = df_to_fit.loc[df_to_fit.doy > df_to_fit.summer_peak]
    #%%
    
    try:
        cor_wb = cor_skipna(dfgpp.waterbal,dfgpp.cond)
        cor_si = cor_skipna(dfgpp.sinterp,dfgpp.cond)
        
        #if cor_si[0] > cor_wb[0]:
        dfgpp["waterbal"] = 1*dfgpp.sinterp
    
    except:
        continue
    #%%
    #dfgpp.kgpp = np.median(dfgpp.kgpp)
    
    # #%%
    dfgpp = dfgpp.loc[dfgpp.waterbal < 0].copy()
    #dfgpp = dfgpp.loc[dfgpp.sinterp < 0].copy()

#%%
    dfi = fit_tau(dfgpp.copy())#.copy()
    
    #%%
    # dfi2 = dfi.copy()
    # dfi2["vpd"] = np.mean(dfi2.vpd)
    # dfi3 = fit_tau(dfi2.copy())#.copy()

    #%%
    #if cor_skipna(dfi.waterbal,dfi.g_adj).pvalue > 0.05:
    #    continue
    #%%
    # m1 = smf.ols("g_adj ~ waterbal + C(year)", data=dfi).fit()
    # #%%
    # offsets = (m1.predict(dfi) - m1.params[-1]*dfi.waterbal)/m1.params[-1]
    # #%%
    # dfi2 = dfi.copy()
    # dfi2.waterbal = dfi.waterbal+offsets

    # #%%
    # dfi3 = fit_tau(dfi2.copy())#.copy()
    
    # dfi["tauO"] = dfi3.tau.iloc[0]
    # dfi["etr2_O"] =  dfi3.etr2_smc.iloc[0]
    # mod0 = smf.ols("g_adj ~ waterbal",data=dfi).fit()
    # mod0y = smf.ols("g_adj ~ C(year)",data=dfi).fit()

    # mod1 = smf.ols("g_adj ~ waterbal + C(year)",data=dfi).fit()
    # if mod0.rsquared < mod0y.rsquared:
    #     print("strong year effect")
    #     continue
    # dfi["m0_r2"] = mod0.rsquared
    # dfi["m1_r2"] = mod1.rsquared

    #%%
    #dfi = fit_tau_mm(dfgpp).copy()
    #%%
#     nbs = 50
#     bs_tau = np.zeros(nbs)
#     bs_etr2_null = np.zeros(nbs)
#     bs_etr2_smc = np.zeros(nbs)

#     for jbs in range(nbs):
#         #rsel = df_to_fit.sample(len(df_to_fit),replace=True)
#         #dfgpp_samp = fit_gpp(rsel,0)
#         dfgpp_samp = dfgpp.sample(len(dfgpp),replace=True)
#         dfi_samp = fit_tau(dfgpp_samp)
#         bs_tau[jbs] = dfi_samp.tau.iloc[0]
#         bs_etr2_smc[jbs] = dfi_samp.etr2_smc.iloc[0]
#         bs_etr2_null[jbs] = dfi_samp.etr2_null.iloc[0]
# #%%
        
#     dfi["tau_25"] = np.quantile(bs_tau,0.25)
#     dfi["tau_75"] = np.quantile(bs_tau,0.75)
#     dfi["tau_std"] = np.std(bs_tau)
    
    #%%
    # if dfi.etr2_smc.iloc[0] > dfi.etr2_null.iloc[0] + 0.01:
    
    # else:
    #     dfi["tau_25"] = np.nan
    #     dfi["tau_75"] = np.nan
#%%
    # try:
    #     si1 = np.sum(np.isfinite(dfgpp.sinterp))
    #     if si1 > 25:
    #         dfgpp["waterbal"] = 1*dfgpp.sinterp
    #         dfi2 = fit_tau(dfgpp).copy()
            
    #         if dfi2.etr2_smc.iloc[0] > dfi.etr2_smc.iloc[0]:
    #             dfi = dfi2.copy()
    #     else:
    #         print("No soil moisture")
    #         continue
    # except AttributeError:
    #     print("No soil moisture")
    #     continue
    
    #%%
    # df2 = dfi.loc[dfi.g_adj < np.median(dfi.g_adj)].copy()
    # df2["g_adj2"] = (df2["g_adj"]*1000)**2
    # modW = smf.ols("waterbal ~ g_adj2",data=df2).fit()
    # modWY = smf.ols("waterbal ~ 0 + g_adj2 + C(year)",data=df2).fit()
    # #%%
    # dfi["c1"] = modW.params[-1]
    # dfi["c2"] = modWY.params[-1]
    
    # if modW.rsquared/modWY.rsquared < 0.5:
    #     print("too much IAV in soil")
    #     continue
    #%%
    
    #%%
    # yearlist = pd.unique(dfgpp.year)
    
    # # #wrange_list = np.floor(dfi.waterbal / (np.min(dfi.waterbal)-1)*4)
    # # #wlen = 50
    # cdata = []
    # for yi in yearlist:
    # # #for qi in range(4):
    # # #for wi in np.arange(0,len(dfgpp)-wlen,wlen):
    # #     #dfy = df_to_fit.loc[df_to_fit.year==yi].copy()
    # #     #dfg_year, gres_year = fit_gpp(dfy,0,gppres.x[0],gppres.x[1])
    # #     #dfi_year = fit_tau_mm(dfg_year)
    # #     #dfg_year = dfgpp.iloc[wi:(wi+wlen)].copy()
    #     dfg_year = dfgpp.loc[dfgpp.year==yi].copy()
    # #     #dfg_year = dfgpp.loc[wrange_list==qi].copy()  
    #     if len(dfg_year) >= 10:
    #         dfi_year = fit_tau(dfg_year)        
    #         cdata.append(dfi_year)
    # samp_all = pd.concat(cdata)
    # # #%%
    # samp_good = samp_all[samp_all.etr2_smc > 0]
    # samp_good = samp_good[samp_good.etr2_smc > samp_good.etr2_null+0.05].reset_index()
    # #%%
    # if len(samp_good) == 0:
    #     continue
    # # if len(samp_good) > 10:
    # #     dfi_good = fit_tau(samp_good.copy())
    # #     dfi["tau_goodyear"] = dfi_good.tau.iloc[0]
    # # else:
    # #     dfi["tau_goodyear"] = np.nan
        
    # # year_tau_dict[site_id] = pd.unique(samp_good.tau)
    # #%%
    # samp_good2 = samp_good.copy()
    # samp_good2.waterbal = (samp_good2.waterbal/1000 - samp_good2.smin)*1000
    # dfiS2 = fit_tau(samp_good2)
    
    # dfi["tau_year"] = dfiS2.tau.iloc[0]
    # dfi["etr2_smc_year"] = dfiS2.etr2_smc.iloc[0]
    # dfi["etr2_null_year"] = dfiS2.etr2_null.iloc[0]

    # #dfi["quant_tau"] = np.mean(samp_good.tau)
    # #%%
    # dfiS1 = fit_tau(samp_good)
#%%
#    dfi = fit_tau(dfgpp)

    # dfi_early = fit_tau(dfgpp.loc[dfgpp.doy <= mid_day].copy())
    # dfi_late = fit_tau(dfgpp.loc[dfgpp.doy > mid_day].copy())
    # dfi["tau_early"] = dfi_early.tau.iloc[0]
    # dfi["tau_late"] = dfi_late.tau.iloc[0]
    
    # mid_water = (np.max(dfgpp.waterbal) + np.min(dfgpp.waterbal))/2
    # dfi_hiwat = fit_tau(dfgpp.loc[dfgpp.waterbal <= mid_water].copy())
    # dfi_lowat = fit_tau(dfgpp.loc[dfgpp.waterbal > mid_water].copy())
    # dfi["tau_hiwat"] = dfi_hiwat.tau.iloc[0]
    # dfi["tau_lowat"] = dfi_lowat.tau.iloc[0]
    

    # g_n = dfgpp_mm.cond/dfgpp_mm.kgpp
    # gn2 = g_n - np.log(g_n+1)
    # lhs_gg = gn2*2*dfgpp_mm.kgpp*(dfgpp_mm.vpd/100)/zsoil_mol
    # invmod_mm = sm.OLS(lhs_gg,sm.add_constant(dfgpp_mm.waterbal/1000)).fit()

    # dfi["tau_mm"] = 1/invmod_mm.params[1]/(60*60*24)
    # dfi["smin_mm"] = -invmod_mm.params[0]/invmod_mm.params[1]
    # #dfi["gslope_mm"] = dfgpp_mm.gpp_slope.iloc[0]
    # dfi["gppmax_mm"] = dfgpp_mm.gppmax
    #%%
    #dfgppY = fit_gpp(df_to_fit,1)
    #dfiY = fit_tau(dfgppY)
#%%
    dfi["npoints"] = len(dfi)
    all_results.append(dfi)
#%%
all_results = pd.concat(all_results)
#%%
site_count = np.array(all_results.groupby("SITE_ID").count()["waterbal"])
site_year = np.array(all_results.groupby("SITE_ID").nunique()["year"])

#%%
df1 = all_results.groupby("SITE_ID").first().reset_index()
df1["site_count"] = site_count
df1["year_count"] = site_year

df1["Aridity"] = df1.mean_netrad / (df1.map_data / (18/1000 * 60*60*24) * 44200)
#df1["Aridity_gs"] = df1.gs_netrad / (df1.mgsp_data / (18/1000 * 60*60*24) * 44200)
#f_stat = (df1.etr2_smc-df1.etr2_null)/(1-df1.etr2_smc)*(df1.site_count-2)
#scipy.stats.f.cdf(f_stat,1,df1.site_count-2)
#%%

# #first account for the ones that have ET

# for x in pd.unique(df1.SITE_ID):
#     site_data = df1.loc[df1.SITE_ID==x].iloc[0]
#     if site_data.mat_data <= 3:
#         message = "Mean temperature < 3 C"
#     #elif site_data.gppR2 <= 0:
#     #    message = "GPP model did not fit"
#     elif site_data.etr2_smc <= 0:
#         message = "Conductance model did not fit"
#     elif (1-df1.etr2_smc)/(1-df1.etr2_null) >= 0.9:
#     #elif (1-site_data.etr2_smc) / (1-site_data.etr2_null) >= 0.9:
#         message = "Not water limited"
#     else:
#         message = "Water limited"
#     site_result[x] = message
#%%

df1 = df1.loc[df1.etr2_smc > 0]
#df1 = df1.loc[df1.gppR2 > 0]
#%%
#df1 = df1.loc[df1.etr2_smc-df1.etr2_null > 0.01]
#df1 = df1.loc[(1-df1.etr2_smc)/(1-df1.etr2_null) < 0.9]


#%%
#df1 = df1.loc[df1.SITE_ID != "AU-Emr"]
#df1 = df1.loc[df1.SITE_ID != "AU-How"]

#df1 = df1.loc[df1.site_count > 100]
#df1 = df1.loc[df1.year_count >= 2]
df1 = df1.loc[df1.mat_data > 3]
#%%
df1 = pd.merge(df1,bif_forest,on="SITE_ID",how="left")
#%%
def qt_gt1(x,q):
    return np.quantile(x[x >= 1],q)
def mean_gt1(x):
    return np.mean(x[x >= 1])
#%%
df_meta = df1.copy()

#%%
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
df_meta["combined_biome"] = [simple_biomes[x] for x in df_meta["IGBP"]]
#df_meta = df_meta.loc[df_meta.gppR2 > df_meta.gppR2_null]
#df_meta = df_meta.loc[df_meta.gppR2 > df_meta.gppR2_cond_only]
#df_meta = df_meta.loc[df_meta.m0_r2/df_meta.m1_r2 > 0.5].copy()
#%%
def get_lens(x,c):
    x2 = 1*x
    x2[0] = c+1
    day_diff = np.diff(np.where(x2 > c)[0])
    return day_diff[day_diff >= 1]

# ddl_rain = {}
# for x in df_meta.SITE_ID:
#     rainX = rain_dict[x][0]
#     ddl_rain[x] = get_lens(rainX,10)
def interval_len(x,c):
    acc = 0
    tip_list = [0]
    for j in range(len(x)):
        acc += x[j]
        if acc >= c:
            acc = 0
            tip_list.append(j)
    day_diff = np.diff(np.array(tip_list))
    return day_diff
#%%
ddl_rain = {}
rain_gs_dict = {}
for x in df_meta.SITE_ID:
    rain_site = rain_data.loc[rain_data.SITE_ID==x].copy()
    rain_allyear = np.array(rain_site.rain_mm)
    year_list = np.array(rain_site.year)
    rain_gs_dict[x] = np.mean(rain_allyear[np.isfinite(rain_allyear)])
    years_max = []
    for y in np.unique(year_list):
        #cutoff = df_meta.map_data.loc[df_meta.SITE_ID==x].iloc[0]*4
        years_max.append(np.max(get_lens(rain_allyear[year_list==y],10)))
        #years_max.append(np.max(interval_len(rain_allyear[year_list==y],20)))

#        years_max.append(np.mean(get_lens(rain_allyear[year_list==y],10)))
        
    ddl_rain[x] = years_max
#%%
# #%%
# df_meta["gs_len"] = df_meta.gs_end-df_meta.gs_peak
# #non_season_limited = df_meta.loc[df_meta.tau < 0.9*df_meta["gs_len"]]
#%%
#df_meta["rain_pt"] = [prob_stay_dry(rain_dict[x],5, 5) for i,x in enumerate(df_meta.SITE_ID)]
df_meta["ddrain_mean"] = [np.mean(ddl_rain[x]) for i,x in enumerate(df_meta.SITE_ID)]
#df_meta["ddrain_95"] = [np.quantile(ddl_rain[x],0.95) for i,x in enumerate(df_meta.SITE_ID)]
#df_meta["ddrain_max"] = [np.max(ddl_rain[x]) for i,x in enumerate(df_meta.SITE_ID)]
#df_meta["ddrain_90"] = [np.quantile(ddl_rain[x],0.90) for i,x in enumerate(df_meta.SITE_ID)]
df_meta["gsrain_mean"] = [rain_gs_dict[x] for x in df_meta.SITE_ID]
#%%
#df_meta["tau_rel_unc"] = (df_meta.tau_75-df_meta.tau_25)/df_meta.tau
#%%
fval = ((1-df_meta.etr2_null)-(1-df_meta.etr2_smc))/(1-df_meta.etr2_smc)*(df_meta.npoints-4)
df_meta["ftest"] = 1-scipy.stats.f.cdf(x=fval,dfn=1,dfd=df_meta.npoints-4)
#df_meta = df_meta.loc[ftest > 0.99]
#df_meta = df_meta.loc[df_meta.tau_rel_unc < 0.25].copy()
#%%
df_meta = df_meta.loc[df_meta.gppR2 > 0.0].copy()
df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_no_cond > 0.0]
df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_only_cond > 0.0]
df_meta = df_meta.loc[df_meta.smin > -10]
#%%
df_meta = df_meta.loc[df_meta.tau > 0]
#df_meta = df_meta.loc[df_meta.tau_lo > 0]
#df_meta = df_meta.loc[df_meta.tau_hi > 0]

#%%
df_meta["rel_err"] = (df_meta.etr2_smc-df_meta.etr2_null)/(1-df_meta.etr2_null)
df_meta = df_meta.loc[df_meta.rel_err > 0.1]
#%%
resmean = all_results.groupby("SITE_ID").mean(numeric_only=True).reset_index()
df_meta = pd.merge(df_meta,resmean[["SITE_ID","kgpp","gpp"]],how='left',on='SITE_ID')
#df_meta = df_meta.loc[df_meta.summer_end - df_meta.summer_start - df_meta.tau > 0]
#%% 
rainmod = smf.ols("tau ~ ddrain_mean",data=df_meta).fit()
rainmodW = smf.wls("tau ~ ddrain_mean",data=df_meta,weights=df_meta.etr2_smc).fit()
#rainmod_gpp_add = smf.ols("tau ~ ddrain_mean",data=df_meta).fit()
#rainmod_gpp2 = smf.ols("tau ~ ddrain_mean + kgpp_y",data=df_meta).fit()

#%%
fig,ax = plt.subplots(1,1,figsize=(10,8))

lmax = 500

line1, = ax.plot([0,lmax],[0,lmax],"k",label="1:1 line")
line2, = ax.plot([0,lmax],np.array([0,lmax])*rainmod.params[1]+rainmod.params[0],"b--",label="Regression line\n($R^2$ = 0.52)")
#plt.plot([0,150],np.array([0,150])*reg0.params[0],"b--",label="Regression line\n($R^2$ = 0.39)")
#leg1 = ax.legend(loc="upper left")
leg1 = ax.legend(loc="lower right")

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.ddrain_mean,subI.tau,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

ax.set_xlim(0,150)
ax.set_ylim(0,150)
ax.set_xlabel("Annual-mean longest dry period (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")

#%%
water_limitation = pd.DataFrame({"SITE_ID":site_result.keys(),
                                 "Results":site_result.values()}).sort_values("SITE_ID")
#%%
all_results["cond_water_limited"] = all_results["pred_cond"] < all_results["gmax"]
#all_results["frac_water_limited"] = all_results["pred_cond"] / all_results["gmax"]

#all_results["ET_water_limited"] = all_results["et_tau"] < all_results["et_null"]
amean = all_results.groupby("SITE_ID").mean(numeric_only=True).reset_index()
#%%
df_meta = pd.merge(df_meta,amean[["SITE_ID","cond_water_limited"]],how="left",on="SITE_ID")
#%%
df_meta["DOM_DIST_MGMT"] = df_meta["DOM_DIST_MGMT"].fillna("None")
#%%
# smin = dfi.smin.iloc[0]
# s_adj = dfi.waterbal/1000 - smin
# slope = np.sqrt(1/(dfi.tau.iloc[0]*(60*60*24)))

# #final_cond = np.clip(slope*np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100)),gmin,gmax)
# xarr = np.linspace(smin,np.max(dfi.waterbal/1000),500)

# plt.figure()
# plt.plot(dfi.waterbal/1000,dfi.g_adj,'o')
# plt.plot(xarr,slope*np.sqrt(np.clip(xarr-smin,0,dfi.width.iloc[0])))
#%%%%
# in each patch, A(g) = L*(1-exp(-g/L*k))
# and g ~ sqrt(L)
# Two patches acting independently
# A(g) = A((g1+g2)/2)/2 = (L*(1-exp(-g1/L*k)) + L*(1-exp(-g2/L*k)))/2
# A(g) = L*
#%%
# g95 = []
# a95 = []

# for x in pd.unique(all_results.SITE_ID):
#     data_site = all_results.loc[all_results.SITE_ID==x].copy()
#     higpp = data_site.loc[data_site.gpp > np.quantile(data_site.gpp,0.95)]
#     g95.append(np.median(higpp.cond))
#     a95.append(np.median(higpp.gpp))
    #%%
plt.figure()
plt.plot(df_meta.gpp_y,rainmod.resid,'o')
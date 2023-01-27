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
bif_data = pd.read_csv("fn2015_bif_tab.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])]
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])]
#metadata = pd.read_csv(r"C:\Users\nholtzma\Downloads\fluxnet_site_info_all.csv")

all_daily = glob.glob("daily_data\*.csv")
forest_daily = [x for x in all_daily if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
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
#filename_list = glob.glob("processed_nov7b/*.csv")
#%%
from modular_analysis_functions import prepare_df, fit_gpp, fit_tau
#%%
#%%
#width = 1
zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf

#%%
#dd_dict1 = {}
#dd_dict_soil = {}
#dd_dict_soil_smooth = {}
#dd_dict_wbal_smooth = {}
rain_dict = {}
#wb_dict = {}
tau_by_year = {}

all_results = []
site_result = {}
#%%
for fname in forest_daily:#[forest_daily[x] for x in [70,76]]:
#%%
    site_id = fname.split("\\")[-1].split('_')[1]
    print(site_id)
    df_res = prepare_df(fname, site_id, bif_forest)
    #%%
    if type(df_res) == str:
        site_result[site_id] = df_res
        continue
    #%%
    df_to_fit, rain_res, maT, maP, df_full = df_res
    rain_dict[site_id] = rain_res
    #%%
    if len(df_to_fit) < 25 or sum(np.isfinite(df_to_fit["gpp_qc"])) < 25:
        print("Not enough data")
        site_result[site_id] = "Not enough data in growing season"
        continue
    #%%
    ######
    #bootstrap goes here
    ######
    #%%
    df_to_fit["cond_dpar"] = df_to_fit.cond/df_to_fit.par
    df_to_fit["gpp_dpar"] = df_to_fit.gpp_qc/df_to_fit.par
    
    medcond = (np.min(df_to_fit["cond_dpar"]) + np.max(df_to_fit["cond_dpar"])) / 2
    df_to_fit["shift_cond"] = np.clip(df_to_fit["cond_dpar"]-medcond,0,100)
    greg = smf.ols("gpp_dpar ~ cond_dpar + shift_cond",data=df_to_fit,missing="drop").fit()
    #%%
    if greg.pvalues[2] > 0.05:
        print("GPP model did not fit")
        site_result[site_id] = "GPP model did not fit"
        continue    
    #%%
    dfgpp, gpp_r2, gpp_r2_null = fit_gpp(df_to_fit)
    #%%
    # gpp_res = dfgpp.gpp_pred-dfgpp.gpp_qc
    # good_gpp = np.abs(gpp_res) < 1.5
    # #%%
    # dfgpp, gpp_r2, gpp_r2_null = fit_gpp(df_to_fit.loc[good_gpp])
    
    #%%
    if np.min(dfgpp.gppmax) <= 0:
        print("GPP model did not fit")
        site_result[site_id] = "GPP model did not fit"
        continue    
    #%%
    if (1-gpp_r2)/(1-gpp_r2_null) > 0.9:
        print("No GPP limitation")
        site_result[site_id] = "GPP model did not fit"
        continue
    #plt.figure()
    #plt.plot(df_to_fit.doy,df_to_fit.gpp-gpp_pred,'.')
    #%%
    # df_to_fit["res_gpp"] = df_to_fit.gpp-gpp_pred
    # df_fit_doy = df_to_fit.groupby("doy").mean(numeric_only=True).rename(columns={"res_gpp":"doy_res"})
    # df_to_fit = pd.merge(df_to_fit,df_fit_doy["doy_res"],on="doy",how="left")

    #%%
#    dfgpp.cond = dfgpp.cond - dfgpp.res_cond
#    dfgpp.ET = (dfgpp.cond - dfgpp.res_cond)*dfgpp.vpd/100


    #%%
    #dfgpp.waterbal = 1*dfgpp.sinterp    #1*dfgpp.waterbal_smc
    #%%
    cmin = np.min(dfgpp.waterbal)
    #wsel = (dfgpp.waterbal > -1000)# (dfgpp.waterbal > -500)*(dfgpp.waterbal < cmin/3)
    dfi = fit_tau(dfgpp)#.loc[wsel])
    #%%
#     bs_tau = []
#     for jbs in range(50):
#         rsel = dfgpp.sample(len(dfgpp),replace=True)
#         dfi = fit_tau(rsel)
#         bs_tau.append(dfi.tau.iloc[0])
# #%%
    #good_et = np.abs(dfi.et_tau-dfi.ET)*18/1000*60*60*24 < 1
    #dfi = fit_tau(dfgpp.loc[good_et])
    #%%
    #dfi = fit_tau(dfgpp.loc[dfgpp.year == 2012])

    #%%
    # wsize = 15
    # conc_samp = []

    # for istart in range(0,len(dfgpp)-wsize,wsize):
    #     dfsamp = dfgpp.iloc[istart:istart+wsize]

    # #for yearI in pd.unique(dfgpp.year):
    # #    dfsamp = dfgpp.loc[dfgpp.year==yearI]

    #     dfiS = fit_tau(dfsamp)
    #     conc_samp.append(dfiS)
    # samp_all = pd.concat(conc_samp)
    # #%%
    # samp_good = samp_all[samp_all.etr2_smc > 0]
    # samp_good = samp_good[samp_good.etr2_smc > samp_good.etr2_null+0.1].reset_index()
    # #%%
    # nyear_good = len(pd.unique(samp_good.year))
    # tau_by_year[site_id] = pd.unique(samp_good.tau)

    # #%%
    # if nyear_good < 1:
    #     print("Not enough water stressed years")
    #     site_result[site_id] = "Only one water-limited year"
    #     continue
    # #%%
    # samp_good2 = samp_good.copy()
    # samp_good2.waterbal = (samp_good2.waterbal/1000 - samp_good2.smin)*1000
    # dfiS2 = fit_tau(samp_good2)
#%%
    dfi["SITE_ID"] = site_id
    #dfi["random_effect_tau"] = dfiS2.tau.iloc[0]
    dfi["mat_data"] = maT
    dfi["map_data"] = maP
    dfi["min_smc_sw"] = np.min(dfi.waterbal/1000-dfi.smin)
    
    #dfi["break_tau"] = np.mean(samp_good.tau)
    #dfi["mgsp_data"] = np.nanmean(p_in[late_summer])
    #dfi["mean_netrad"] = np.nanmean(myrn)
    #dfi["mean_netrad"] = np.nanmean(myrn[late_summer])

    #dfi["gs_netrad"] = np.nanmean(myrn[is_late_summer])
    #dfi["pet_year"] = np.nanmean(pet)
    #dfi["pet_late_summer"] = np.nanmean(pet[is_late_summer])

    #%%
    #dfi["snow_frac"] = np.sum(p_in*(airt_summer < 0))/np.sum(p_in)

    #dfi["gs_peak"] = topday
    #dfi["gs_end"] = summer_end
    #plt.plot(np.sort(ddlen[ddlen > 0]),np.linspace(0,1,len(ddlen[ddlen > 0]))); plt.plot(np.arange(80),1-np.exp(-np.arange(80)/10.5))
    #%%

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

#df1["Aridity"] = df1.mean_netrad / (df1.map_data / (18/1000 * 60*60*24) * 44200)
#df1["Aridity_gs"] = df1.gs_netrad / (df1.mgsp_data / (18/1000 * 60*60*24) * 44200)
#f_stat = (df1.etr2_smc-df1.etr2_null)/(1-df1.etr2_smc)*(df1.site_count-2)
#scipy.stats.f.cdf(f_stat,1,df1.site_count-2)
#%%
for x in pd.unique(df1.SITE_ID):
    site_data = df1.loc[df1.SITE_ID==x].iloc[0]
    if site_data.mat_data <= 3:
        message = "Mean temperature < 3 C"
    elif site_data.gppR2 <= 0:
        message = "GPP model did not fit"
    elif site_data.etr2_smc <= 0:
        message = "Conductance model did not fit"
    elif site_data.etr2_smc - site_data.etr2_null <= 0.1:
    #elif (1-site_data.etr2_smc) / (1-site_data.etr2_null) >= 0.9:
        message = "Not water limited"
    else:
        message = "Water limited"
    site_result[x] = message
#%%
#df1 = df1.loc[df1.etr2_smc-df1.etr2_null > 0.1]
df1 = df1.loc[(1-df1.etr2_smc)/(1-df1.etr2_null) < 0.9]


df1 = df1.loc[df1.etr2_smc > 0]
df1 = df1.loc[df1.gppR2 > 0]
#df1 = df1.loc[df1.site_count > 100]
df1 = df1.loc[df1.year_count >= 2]
df1 = df1.loc[df1.mat_data > 3]

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
#%%
# fig,ax = plt.subplots(1,1,figsize=(10,8))

# for i in range(len(biome_list)):
#     subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
#     ax.plot(subI.mat_data,subI.map_data*365,'o',alpha=0.75,markersize=15,label=biome_list[i],color=mpl.colormaps["tab10"](i+2))
# ax.set_xlabel("Mean annual temperature $(^oC$)",fontsize=24)
# ax.set_ylabel("Mean annual precipitation (mm)",fontsize=24)
# #plt.legend(loc="center left",bbox_to_anchor=(1,0.5),ncols=2)
# fig.legend(loc="upper center",bbox_to_anchor=(0.46,0),ncols=2 )

#%%
plot_gadj = 0

if plot_gadj:
    yscaler = np.sqrt(zsoil_mol)
    molm2_to_mm = 18/1000
    s2day = 60*60*24
    plt.figure()

    for site_id in ["IT-Lav","US-MMS"]:
    
        df2 = all_results.loc[all_results.SITE_ID==site_id]
        tauI = df2.tau.iloc[0]
        site_legend = site_id+r", $\tau$ = "+str(np.round(tauI,0))+" days"
        plt.plot((df2.waterbal/1000)*zsoil_mol*molm2_to_mm,
                 (df2.g_adj*yscaler*np.sqrt(molm2_to_mm))**2*s2day,
                 'o',alpha=1,label=site_legend)
        xarr = np.linspace(0,1000,500)
        #tau_overall = np.median(df2.tau)
        print(tauI)
        #width=df2.width.iloc[0]
        plt.plot(xarr+df2.smin.iloc[0]*zsoil_mol*molm2_to_mm,(np.clip(xarr,0,width)/tauI/(60*60*24))*s2day,'k',linewidth=2)
        
    # tau2 = 40.6
    # smin2 = -0.5
    # plt.plot(xarr+smin2*zsoil_mol*molm2_to_mm,(np.clip(xarr,0,width)/tauI/(60*60*24)),'g',linewidth=3)
    
    plt.xlabel("$Z_{soil}*s$ (mm) (0 = saturation)",fontsize=30)
    #plt.ylabel("Adjusted conductance $(s^{0.5})$")
    plt.ylabel("$g_{adj}$ (mm/day)",fontsize=30)
    
    plt.legend(fontsize=24,framealpha=1)
    
    plt.xlim(-300,0)
    plt.ylim(0,20)
    #plt.title(site_id+", tau 
#%%

#%%
def get_lens(x,c):
    x2 = 1*x
    x2[0] = c+1
    day_diff = np.diff(np.where(x2 > c)[0])
    return day_diff[day_diff >= 1]

ddl_rain = {}
for x in df_meta.SITE_ID:
    rain_allyear = rain_dict[x][0]
    year_list = rain_dict[x][1]

    years_max = []
    for y in np.unique(year_list):
        years_max.append(np.max(get_lens(rain_allyear[year_list==y],10)))
#        years_max.append(np.mean(get_lens(rain_allyear[year_list==y],10)))
        
    ddl_rain[x] = years_max
#%%
# #%%
# df_meta["gs_len"] = df_meta.gs_end-df_meta.gs_peak
# #non_season_limited = df_meta.loc[df_meta.tau < 0.9*df_meta["gs_len"]]
#%%
df_meta["ddrain_mean"] = [np.mean(ddl_rain[x]) for x in df_meta.SITE_ID]
rainmod = smf.ols("tau ~ ddrain_mean",data=df_meta).fit()
#%%
df_meta["rain_amt_mean"] = [np.nanmean(rain_dict[x][0][np.isfinite(rain_dict[x][0])]) for x in df_meta.SITE_ID]
#df_meta#["rain_amt_mean"] = [np.nanmean(rain_dict[x][0][np.isfinite(rain_dict[x][0])*(rain_dict[x][0] >0)]) for x in df_meta.SITE_ID]

#%%
fig,ax = plt.subplots(1,1,figsize=(10,8))

lmax = 500

line1, = ax.plot([0,lmax],[0,lmax],"k",label="1:1 line")
line2, = ax.plot([0,lmax],np.array([0,lmax])*rainmod.params[1]+rainmod.params[0],"b--",label="Regression line\n($R^2$ = 0.58)")
#plt.plot([0,150],np.array([0,150])*reg0.params[0],"b--",label="Regression line\n($R^2$ = 0.39)")
leg1 = ax.legend(loc="upper left")

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    pointI, = ax.plot(subI.ddrain_mean,subI.tau,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
    points_handles.append(pointI)

plt.xlim(0,150)
plt.ylim(0,150)
ax.set_xlabel("Annual-mean longest dry period (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")

#%%
water_limitation = pd.DataFrame({"SITE_ID":site_result.keys(),
                                 "Results":site_result.values()}).sort_values("SITE_ID")
#%%
import cartopy.crs as ccrs
import cartopy.feature as cf
#%%
fig = plt.figure(figsize=(15,15),dpi=100)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.stock_img()
ax.add_feature(cf.LAKES)
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
ax.plot(df1.LOCATION_LONG,df1.LOCATION_LAT,'o',alpha=0.75,color="red",markersize=5)
#%%
good_res = all_results.loc[all_results.SITE_ID.isin(df_meta.SITE_ID)].reset_index()
good_res = pd.merge(good_res,df_meta[["SITE_ID","ddrain_mean"]],on="SITE_ID",how="left")

#%%
laidata = pd.read_csv("fluxnet_lai_mean4yr.csv")
all_results = pd.merge(all_results,laidata,on="SITE_ID")
all_results["LAImax"] /= 1000
#all_results["LAImax"]  = np.clip(all_results["LAImax"],0,4)
#%%
all_results["gm_dpar"] = all_results.gppmax/all_results.par
resmax = all_results.groupby("SITE_ID").mean(numeric_only=True).reset_index()
#%%
plt.figure()
plt.plot(resmax.LAImax,resmax.gm_dpar,'o')
plt.xlim(0,10)
plt.ylim(0,0.1)
#%%
all_results["gppmax_lai"] = all_results.LAImax*0.00625*all_results.par
#%%
all_results["gpp_pred_lai"] = all_results["gppmax_lai"]*(1-np.exp(-all_results.cond/all_results["gppmax_lai"]*87))
#%%
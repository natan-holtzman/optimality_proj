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
from functions_from_nov16 import prepare_df, fit_gpp, fit_tau

#%%
#%%
#width = 1
zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf
#%%
dd_dict1 = {}
dd_dict_soil = {}
dd_dict_soil_smooth = {}
dd_dict_wbal_smooth = {}
rain_dict = {}
wb_dict = {}

all_results = []
site_result = {}
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
    if len(df_to_fit) < 25:
        print("Not enough data")
        site_result[site_id] = "Not enough data in growing season"

        continue
    #%%
    if sum(np.isfinite(df_to_fit.gpp_qc)) < 25:
        print("Not enough data")

        site_result[site_id] = "Not enough data in growing season"
        continue
    #%%

    #%%
    #dfi = df_to_fit.loc[df_to_fit.waterbal < np.nanmedian(df_to_fit.waterbal)].copy()

   # dfi = df_to_fit.loc[(df_to_fit.doy >= topday)].copy()
    dfgpp = fit_gpp(df_to_fit)
    dfi = fit_tau(dfgpp)

    #%%
   
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

df1["Aridity"] = df1.mean_netrad / (df1.map_data / (18/1000 * 60*60*24) * 44200)
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
df1 = df1.loc[df1.etr2_smc-df1.etr2_null > 0.1]
#df1 = df1.loc[(1-df1.etr2_smc)/(1-df1.etr2_null) < 0.9]


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
def get_lens(x,c):
    x2 = 1*x
    x2[0] = c+1
    day_diff = np.diff(np.where(x2 > c)[0])
    return day_diff[day_diff >= 1]

# ddl_rain = {}
# for x in df_meta.SITE_ID:
#     rainX = rain_dict[x][0]
#     ddl_rain[x] = get_lens(rainX,10)
    

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
#df_meta["rain_pt"] = [prob_stay_dry(rain_dict[x],5, 5) for i,x in enumerate(df_meta.SITE_ID)]
df_meta["ddrain_mean"] = [np.mean(ddl_rain[x]) for i,x in enumerate(df_meta.SITE_ID)]
df_meta["ddrain_95"] = [np.quantile(ddl_rain[x],0.95) for i,x in enumerate(df_meta.SITE_ID)]
df_meta["ddrain_max"] = [np.max(ddl_rain[x]) for i,x in enumerate(df_meta.SITE_ID)]
df_meta["ddrain_90"] = [np.quantile(ddl_rain[x],0.90) for i,x in enumerate(df_meta.SITE_ID)]

#%%
rainmod = smf.ols("tau ~ ddrain_mean",data=df_meta).fit()

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

plt.xlim(0,130)
plt.ylim(0,130)
ax.set_xlabel("Annual-mean longest dry period (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")

#%%
water_limitation = pd.DataFrame({"SITE_ID":site_result.keys(),
                                 "Results":site_result.values()}).sort_values("SITE_ID")

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
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])].copy()
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])].copy()
metadata = pd.read_csv("fluxnet_site_info_all.csv")


# simple_biomes = {"SAV":"Savanna/shrub",
#                  "WSA":"Savanna/shrub",
#                  "CSH":"Savanna/shrub",
#                  "OSH":"Savanna/shrub",
#               "EBF":"Broadleaf/mixed forest",
#               "ENF":"Needleleaf forest",
#               "DNF":"Needleleaf forest",

#               "GRA":"Grassland",
#               "DBF":"Broadleaf/mixed forest",
#               "MF":"Broadleaf/mixed forest",
#               }
# # biome_list = ["Evergreen needleleaf forest", "Mixed forest", "Deciduous broadleaf forest", "Evergreen broadleaf forest",
# #               "Grassland","Shrubland","Savanna"]
# #%%
# color_dict = {"Savanna/shrub":"pink",
#               "Broadleaf/mixed forest":"blue",
#               "Needleleaf forest":"green",
#               "Grassland":"red"}

# #%%
# bif_forest["combined_biome"] = [simple_biomes[x] for x in bif_forest["IGBP"]]
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
rain_data = pd.read_csv("rain_50_50_include_unbalance3.csv")

rain_sites = pd.unique(rain_data.SITE_ID)
ddl_rain = []
rain_gs_mean = []
for x in rain_sites:
    rain_site = rain_data.loc[rain_data.SITE_ID==x].copy()
    rain_allyear = np.array(rain_site.rain_mm)
    year_list = np.array(rain_site.year)
    rain_gs_mean.append(np.mean(rain_allyear[np.isfinite(rain_allyear)]))
    years_max = []
    for y in np.unique(year_list):
        #cutoff = df_meta.map_data.loc[df_meta.SITE_ID==x].iloc[0]*4
        z = 1*rain_allyear[year_list==y]
        z[-1] = np.inf
        years_max.append(np.max(get_lens(z,10)))
        #years_max.append(np.max(interval_len(rain_allyear[year_list==y],20)))

#        years_max.append(np.mean(get_lens(rain_allyear[year_list==y],10)))
        
    ddl_rain.append(np.mean(years_max))
#%%
rain_site_tab = pd.DataFrame({"SITE_ID":rain_sites,
                              "ddrain_mean":ddl_rain,
                              "gsrain_mean":rain_gs_mean})
bif_forest = pd.merge(bif_forest,rain_site_tab,on="SITE_ID",how="left")
#%%
evidata = pd.read_csv("meanEVI_pheno.csv")
evidata["EVImax"] = (evidata["EVI_Minimum_1"] + evidata["EVI_Amplitude_1"])/10000
evidata["EVIamp"] = (evidata["EVI_Amplitude_1"])/10000

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

#from double_doy_effect_gmax_gmin2 import fit_tau, fit_tau_width
# from fit_gpp_use_evi import fit_gpp_evi
# from fit_gpp_airt_slope import fit_gpp_setslope
# from simple_gpp_mod import fit_gpp_simple

# from gmax_gmin_doy_mm import fit_gpp_mmY

#from mm_gmax_gmin import fit_tau_mm
from fit_tau_res_cond import fit_tau_res, fit_assume_tau_res, fit_tau_res_width
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
site_result = {}
#%%
#sites_late_season = pd.read_csv("late_summer_data.csv")
#rain_data = pd.read_csv("rain_late_season.csv")
#%%
#sites_late_season = pd.read_csv("gs_50_50_include_unbalance3_baseline.csv")
sites_late_season = pd.read_csv("gs_50_50_include_unbalance5.csv")


#sites_late_season["gpp"] = (sites_late_season["gpp_qc"] + sites_late_season["gpp_nt"])/2 

sites_late_season = sites_late_season.loc[sites_late_season["gpp"]  > 0]

#sites_late_season["gpp_frac"] = sites_late_season.gpp_pred/sites_late_season.gppmax

site_max = sites_late_season.groupby("SITE_ID").max(numeric_only=True).reset_index()
site_min = sites_late_season.groupby("SITE_ID").min(numeric_only=True).reset_index()

#%%

sites_late_season = pd.merge(sites_late_season,evidata,how="left",on="SITE_ID")
#sites_late_season = sites_late_season.loc[np.isfinite(sites_late_season.EVImax)]
site_max = sites_late_season.groupby("SITE_ID").max(numeric_only=True).reset_index()

site_max = sites_late_season.groupby("SITE_ID").quantile(0.9,numeric_only=True).reset_index()

#%%
# sites_late_season = pd.merge(sites_late_season,laidata,how="left",on="SITE_ID")
# sites_late_season = sites_late_season.loc[np.isfinite(sites_late_season.LAImax)]
# site_max = sites_late_season.groupby("SITE_ID").max(numeric_only=True).reset_index()
sites_late_season = pd.merge(sites_late_season,bif_forest,on="SITE_ID",how='left')
#%%
#sites_late_season["slope_arr"] = 110

#sites_late_season["slope_arr"] = 70 + 20*(sites_late_season["IGBP"].str.endswith("F"))
hi_cond = sites_late_season.loc[(sites_late_season.par > 500)*(sites_late_season.airt > 15)].copy()


#%%
site_gppmax = {}
site_rescond = {}
#site_slope = {}

for x in pd.unique(hi_cond.SITE_ID):
    dfs = hi_cond.loc[hi_cond.SITE_ID==x]
    if len(dfs) < 10:
        continue
    def to_opt(pars):
        
        gpp_pred = pars[0]*(1-np.exp(-(dfs.cond-max(pars[1],0))/pars[0]*90))
        return gpp_pred-dfs.gpp
    if np.isfinite(dfs.EVImax.iloc[0]):
        max0 = dfs.EVImax.iloc[0]*22
    else:
        max0 = 12
    res0 = 0.02*max0
    fit0 = np.array([max0,res0])
    gpp_optres = scipy.optimize.least_squares(to_opt,x0=fit0,method="lm",x_scale=np.abs(fit0))
    site_gppmax[x] = gpp_optres.x[0]
    site_rescond[x] = max(0,gpp_optres.x[1])
    #site_slope[x] = gpp_optres.x[2]

#%%
# hi_cond["gppmax"] = [site_gppmax[x] for x in hi_cond.SITE_ID]
# hi_cond["res_cond"] = [site_rescond[x] for x in hi_cond.SITE_ID]
# hi_cond["slope_arr"] = [site_slope[x] for x in hi_cond.SITE_ID]

#%%
#plt.figure()
#plt.plot((hi_cond.cond-hi_cond.res_cond)/hi_cond.gppmax,hi_cond.gpp/hi_cond.gppmax,'.',alpha=0.2);
#plt.xlim(0,0.05); plt.ylim(0,1.5); plt.plot(xarr,1-np.exp(-xarr*90));

#%%
sites_late_season = sites_late_season.loc[sites_late_season.SITE_ID.isin(site_gppmax.keys())].copy()
sites_late_season["gppmax"] = [site_gppmax[x] for x in sites_late_season.SITE_ID]
sites_late_season["res_cond"] = [site_rescond[x] for x in sites_late_season.SITE_ID]
sites_late_season["slope_arr"] = 90#[site_slope[x] for x in sites_late_season.SITE_ID]

sites_late_season["cond"] = np.clip(sites_late_season["cond"]-sites_late_season["res_cond"],0, np.inf)
#%%
sites_late_season = sites_late_season.loc[sites_late_season["slope_arr"] > 0]
sites_late_season = sites_late_season.loc[sites_late_season["slope_arr"] < 500].copy()

#%%
sites_late_season["gpp_pred_from_max"] = sites_late_season["gppmax"]*(1-np.exp(-sites_late_season["cond"]/sites_late_season["gppmax"]*90))
#%%
parr = np.linspace(0,800,500);
plt.figure()
plt.plot(sites_late_season.par, sites_late_season.gpp / sites_late_season.gpp_pred_from_max,'.',alpha=0.1); plt.ylim(0,2); plt.plot(parr, (parr/650)**0.5)
#%%
tarr = np.linspace(0,30,500);
plt.figure()
plt.plot(sites_late_season.airt, (sites_late_season.gpp / sites_late_season.gpp_pred_from_max),'.',alpha=0.1); 
plt.ylim(0,2); 
plt.plot(tarr, (tarr/15)**0.5)
#%%
z1 = np.array(sites_late_season.gpp / sites_late_season.gpp_pred_from_max)#,0.25,1.75)
z1[z1 > 1.75] = np.nan
z1[z1 < 0.25] = np.nan

sites_late_season["resid"] = z1
sites_late_season["doy_late_rel"] = np.clip(np.array(sites_late_season.doy-sites_late_season.summer_peak),0,300)/np.array(sites_late_season.summer_end-sites_late_season.summer_peak)
sites_late_season["doy_early_rel"] = -np.clip(np.array(sites_late_season.doy-sites_late_season.summer_peak),-300,0)/np.array(sites_late_season.summer_peak-sites_late_season.summer_start)

sites_late_season["doy_rel_total"] = -sites_late_season.doy_early_rel + sites_late_season.doy_late_rel

m1 = smf.ols("resid ~ np.sqrt(par) + np.clip(airt,0,20)",data=sites_late_season).fit()
#m1 = smf.ols("resid ~ np.sqrt(par) + np.clip(airt,0,20) + doy_early_rel + doy_late_rel",data=sites_late_season).fit()
#%%
m2 = smf.ols("np.log(resid) ~ np.log(par) + np.log(np.clip(airt,1,25))",data=sites_late_season).fit()

#%%
# par_array = np.array(sites_late_season.par)**0.5
# airt_array = np.clip(np.array(sites_late_season.airt),0,20)

par_array = np.array(sites_late_season.par)
airt_array = np.clip(np.array(sites_late_season.airt),1,25)

gmax_array = np.array(sites_late_season.gppmax)
gpp_array = np.array(sites_late_season.gpp)
cond_array = np.array(sites_late_season.cond)

slope_arr = np.array(sites_late_season["slope_arr"])

def to_opt(pars):
#    gmax_mult = pars[0] + pars[1]*par_array + pars[2]*airt_array
    gmax_mult = np.exp(pars[0]) * par_array**pars[1] * airt_array**pars[2]
    slope_mult = np.exp(pars[3]) * airt_array**pars[4]
    gmax_new = gmax_mult*gmax_array
    gpp_pred = gmax_new*(1-np.exp(-cond_array/gmax_new*90*slope_mult))
    return gpp_pred-gpp_array
fit0 = np.array([-2.7,0.3,0.25,-2.7,0.25])
gpp_optres = scipy.optimize.least_squares(to_opt,x0=fit0,method="lm",x_scale=np.abs(fit0))
#%%
#sites_late_season["par_t_fac"] = m1.fittedvalues
pars = gpp_optres.x
#sites_late_season["par_t_fac"] = pars[0] + pars[1]*par_array + pars[2]*airt_array
sites_late_season["par_t_fac"] = np.exp(pars[0]) * par_array**pars[1] * airt_array**pars[2]
sites_late_season["slope_fac"] = np.exp(pars[3]) *  airt_array**pars[4]

#%%
sites_late_season["gppmax2"] = sites_late_season["gppmax"]*sites_late_season["par_t_fac"]
sites_late_season["gpp_pred"] = sites_late_season["gppmax2"]*(1-np.exp(-sites_late_season["cond"]/sites_late_season["gppmax2"]*90*sites_late_season["slope_fac"]))
#%%
plt.figure()
plt.plot(sites_late_season["gpp_pred"], sites_late_season.gpp,'.',alpha=0.1);
plt.plot([0,15],[0,15])
#%%
sites_late_season["kgpp"] = sites_late_season["gppmax2"]/slope_arr/sites_late_season["slope_fac"]
#%%

#mult_array = np.exp(np.linspace(np.log(0.5),np.log(2),50))
#%%
all_results = []

for site_id in pd.unique(sites_late_season.SITE_ID)[:]:#[forest_daily[x] for x in [70,76]]:
#%%
    print(site_id)
    dfgpp = sites_late_season.loc[sites_late_season.SITE_ID==site_id].copy()
    dfgpp = dfgpp.loc[dfgpp.kgpp > 0].copy()
    #%%
    if len(dfgpp) == 0:
        continue
    #%%
    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)
    #dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    #dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2

#%%s
    # dfgpp["waterbal"] = 1*dfgpp.waterbal_x
    dfgpp["waterbal"] = 1*dfgpp.sinterp_anom
    
    #dfgpp["kgpp"] = dfgpp.EVImax*28/100
    
    
    dfi = fit_tau_res(dfgpp.copy())#.copy()
    #%%
    #dfi = fit_tau_res_assume_max(dfgpp.copy(),0.25)#.copy()
    #dfi = fit_tau_res_width(dfgpp.copy())#.copy()

    #%%
    # tau_base = dfi.tau.iloc[0]
    # newr2 = []
    # for tmult in mult_array:
    #     dfi2 = fit_assume_tau_res(dfgpp.copy(),tau_base*tmult)
    #     newr2.append(dfi2.etr2_smc2.iloc[0])
#%%
    dfi["npoints"] = len(dfi)
    all_results.append(dfi)
#%%

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

#%%

df1 = df1.loc[df1.etr2_smc > 0]
#df1 = df1.loc[df1.gppR2 > 0]
#%%
df1 = df1.loc[df1.mat_data > 3]
#%%
#df1 = pd.merge(df1,bif_forest,on="SITE_ID",how="left")
#%%
def qt_gt1(x,q):
    return np.quantile(x[x >= 1],q)
def mean_gt1(x):
    return np.mean(x[x >= 1])
#%%

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
df_meta = df1.copy()

df_meta["combined_biome"] = [simple_biomes[x] for x in df_meta["IGBP"]]

#%%
#df_meta["tau_rel_unc"] = (df_meta.tau_75-df_meta.tau_25)/df_meta.tau
#%%
fval = ((1-df_meta.etr2_null)-(1-df_meta.etr2_smc))/(1-df_meta.etr2_smc)*(df_meta.npoints-4)
df_meta["ftest"] = 1-scipy.stats.f.cdf(x=fval,dfn=1,dfd=df_meta.npoints-4)
#df_meta = df_meta.loc[ftest > 0.99]
#df_meta = df_meta.loc[df_meta.tau_rel_unc < 0.25].copy()
#%%
df_meta = df_meta.loc[df_meta.gppR2 > 0.0].copy()
#%%
#df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_no_cond > 0.0]
#df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_only_cond > 0.0]
df_meta = df_meta.loc[df_meta.smin > -10]
#%%
df_meta = df_meta.loc[df_meta.tau > 0]
#df_meta = df_meta.loc[df_meta.tau_lo > 0]
#df_meta = df_meta.loc[df_meta.tau_hi > 0]

#%%
df_meta["rel_err"] = (df_meta.etr2_smc-df_meta.etr2_null)/(1-df_meta.etr2_null)
#%%
df_meta = df_meta.loc[df_meta.rel_err > 0.1]
#%%
resmean = all_results.groupby("SITE_ID").mean(numeric_only=True).reset_index()
df_meta = pd.merge(df_meta,resmean[["SITE_ID","kgpp","gpp","gpp_pred"]],how='left',on='SITE_ID')
#df_meta = df_meta.loc[df_meta.summer_end - df_meta.summer_start - df_meta.tau > 0]
#%%
aunique = all_results.groupby("SITE_ID").nunique().reset_index().rename(columns={"year":"nyear"})
df_meta = pd.merge(df_meta,aunique[["SITE_ID","nyear"]],how="left",on="SITE_ID")
#df_meta = df_meta.loc[df_meta.nyear >= 3]
#%% 
amin = all_results.groupby("SITE_ID").min(numeric_only=True).reset_index().rename(columns={"waterbal":"wbal_min"})
df_meta = pd.merge(df_meta,amin[["SITE_ID","wbal_min"]],how="left",on="SITE_ID")


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

ax.set_xlim(0,250)
ax.set_ylim(0,250)
ax.set_xlabel("Annual-mean longest dry period (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")
#%%
plt.figure()
plt.plot(df_meta.smin - df_meta.wbal_min/1000, rainmod.resid,'o')
#%%
#plt.figure()
# plt.plot(df_meta.ddrain_mean,df_meta.smin - df_meta.wbal_min/1000, 'o')
# #%%
# all_results2 = []

# for site_id in pd.unique(df_meta.SITE_ID):#[forest_daily[x] for x in [70,76]]:

#     print(site_id)
#     dfgpp = all_results.loc[all_results.SITE_ID==site_id].copy()
#     dfgpp = dfgpp.loc[dfgpp.kgpp > 0].copy()
#     tau_new = dfgpp.ddrain_mean.iloc[0]
#     dfi2 = fit_assume_tau_res(dfgpp.copy(),tau_new)#.copy()
#     all_results2.append(dfi2)
# #%%
# all_results2 = pd.concat(all_results2)
# #%%
# df1b = all_results2.groupby("SITE_ID").first().reset_index()
# #%%
# plt.figure()
# plt.plot(df1b.etr2_smc-df1b.etr2_null,df1b.etr2_smc2-df1b.etr2_null,'o')
# plt.plot([0,0.5],[0,0.5])
#%%
# plt.plot(df1b.etr2_smc-df1b.etr2_null)
# plt.plot(df1b.etr2_smc2-df1b.etr2_null)

#get uncertainty interval by moving tau around, fitting smin, and seeing how r2 changes
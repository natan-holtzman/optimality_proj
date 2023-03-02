# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:46:32 2022

@author: nholtzma
"""

#instead of fitting function of PAR, T
#just use days where PAR, T are high enough


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
#sites_late_season = pd.read_csv("late_summer_data.csv")
#rain_data = pd.read_csv("rain_late_season.csv")
#%%
sites_late_season = pd.read_csv("gs_50_50_mar1.csv")
#sites_late_season.gpp = 1*sites_late_season.gpp_nt #+ sites_late_season.gpp)/2

sites_late_season.gpp = (sites_late_season.gpp_nt + sites_late_season.gpp)/2
#sites_late_season = sites_late_season.loc[sites_late_season.rain_prev==0].copy()
#sites_old = pd.read_csv("plain_fitted_mar1.csv")

#sites_late_season = pd.merge(sites_late_season,sites_old[["SITE_ID","date","sinterp_mean2"]],how="inner",on=["SITE_ID","date"])
#sites_both = pd.merge(sites_late_season,sites_old,how="inner",on=["SITE_ID","date"])

#sites_late_season = sites_late_season.loc[sites_late_season.SITE_ID.isin(pd.unique(sites_old.SITE_ID))]
#%%
sites_late_season = pd.merge(sites_late_season,bif_forest,on="SITE_ID",how='left')
#%%
sites_late_season["res_cond"] = 0


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

sites_late_season["combined_biome"] = [simple_biomes[x] for x in sites_late_season["IGBP"]]

#%%

sites_late_season = sites_late_season.loc[sites_late_season["gpp"]  > 0]
#sites_late_season = sites_late_season.loc[sites_late_season["par"]  > 150]
#%%
sites_late_season["drel_spring"] = -np.clip(sites_late_season["doy"] - sites_late_season["summer_peak"],-np.inf,0) / (sites_late_season["summer_peak"] - sites_late_season["summer_start"])
sites_late_season["drel_fall"] = np.clip(sites_late_season["doy"] - sites_late_season["summer_peak"],0,np.inf) / (sites_late_season["summer_end"] - sites_late_season["summer_peak"])
sites_late_season["drel_both"] = -sites_late_season["drel_spring"] + sites_late_season["drel_fall"]

#%%
xarr_par = np.linspace(0,550,500);
#%%
all_results = []

for site_id in pd.unique(sites_late_season.SITE_ID)[:]:#[forest_daily[x] for x in [70,76]]:
#%%
    print(site_id)
    dfgpp = sites_late_season.loc[sites_late_season.SITE_ID==site_id].copy()
    dfgpp = dfgpp.loc[dfgpp.airt > 5]
    #%%
    cn = 1*dfgpp.cond
    cn -= np.mean(cn)
    dfgpp = dfgpp.loc[np.abs(cn) < 3*np.std(cn)]
    
    #%%
    
    lcn = np.log(dfgpp.cond)
    lcn -= np.mean(lcn)
    dfgpp = dfgpp.loc[np.abs(lcn) < 3*np.std(lcn)]
    
    #lcn = -(dfgpp.waterbal/100)**2
    #lcn -= np.mean(lcn)
    #dfgpp = dfgpp.loc[np.abs(lcn) < 3*np.std(lcn)]
    #%%
    wbs = np.sort(dfgpp.waterbal)
    dsort = np.diff(wbs)
    #%%
    hlen = int(len(dsort)/2)
    if np.max(dsort[:hlen]) > 5:
        wcutoff = wbs[np.where(dsort[:hlen] > 5)[0][-1]]
        dfgpp = dfgpp.loc[dfgpp.waterbal > wcutoff]
    
    #dfgpp = dfgpp.loc[dfgpp.cond > np.quantile(dfgpp.cond,0.1)]
    #dfgpp = dfgpp.loc[dfgpp.LAIint_rel >= 0.75]
    #dfgpp = dfgpp.loc[dfgpp.drel_both >= 0]
    #dfgpp = dfgpp.loc[dfgpp.kgpp > 0].copy()
    #dfgpp = dfgpp.loc[np.isfinite(dfgpp.sinterp_mean2)].copy()
    #%%
    lo_soil = dfgpp.waterbal < np.nanquantile(dfgpp.waterbal,0.25)
    hi_soil = dfgpp.waterbal > np.nanquantile(dfgpp.waterbal,0.75)
    dfgpp["cond_lo_smc"] = np.nanmedian(dfgpp.cond[lo_soil])
    dfgpp["cond_hi_smc"] = np.nanmedian(dfgpp.cond[hi_soil])

    cond_dq = dfgpp.cond * np.sqrt(dfgpp.vpd)
    dfgpp["acond_lo_smc"] = np.nanmedian(cond_dq[lo_soil])
    dfgpp["acond_hi_smc"] = np.nanmedian(cond_dq[hi_soil])
    
    # cond_ld = dfgpp.cond / dfgpp.dayfrac #/ dfgpp.LAIint_rel 
    # dfgpp["bcond_lo_smc"] = np.nanmedian(cond_ld[lo_soil])
    # dfgpp["bcond_hi_smc"] = np.nanmedian(cond_ld[hi_soil])
    
    #%%
    if len(dfgpp) < 10:
        continue
    #%%
    # dfgpp.cond *= 0.6/dfgpp.dayfrac
    # dfgpp.gpp *= 0.6/dfgpp.dayfrac
    # dfgpp.par *= 0.6/dfgpp.dayfrac
    # dfgpp.ET *= 0.6/dfgpp.dayfrac
    
    #%%
    spring_arr = np.abs(np.array(dfgpp.drel_spring))
    fall_arr = np.array(dfgpp.drel_fall)
    # gm_arr = np.array(dfgpp.gppmax)
    gpp_arr = np.array(dfgpp.gpp)
    cond_arr = np.array(dfgpp.cond)
    # k_arr = np.array(dfgpp.kgpp)
    # t_arr = np.array(dfgpp.airt)
    # pr_arr = np.array(dfgpp.potpar)
    par_arr = np.array(dfgpp.par)

    # day_arr = np.array(dfgpp.dayfrac)
    gpp_allmax = np.max(gpp_arr)
 
    #%%
    #%%
    slope0 = 100
    def gpp_opt(pars):
        gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[3]*spring_arr) * (par_arr/275)**pars[2]
        gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
        return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    fit0 = np.array([gpp_allmax,0.1,0.5,0.1])
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    pars = gpp_optres.x
    gppmax = pars[0] * (1 - pars[1]*fall_arr - pars[3]*spring_arr) * (par_arr/275)**pars[2]
    gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    #%%
    # slope0 = 100
    # def gpp_opt(pars):
    #     gppmax = pars[0] * (par_arr/275)**pars[1]
    #     gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    #     return (gpp_pred-gpp_arr)#[np.isfinite(gpp_samp)]
    
    # fit0 = np.array([gpp_allmax,0.5])
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    # pars = gpp_optres.x
    # gppmax = pars[0] * (par_arr/275)**pars[1]
    # gpp_pred = gppmax*(1-np.exp(-cond_arr*slope0/gppmax))
    
    #%%
    if pars[0] < 0:
        continue
    if np.min(gppmax) < 0:
        continue
    #%%
    dfgpp["gppmax"] = gppmax
    dfgpp["kgpp"] = gppmax/slope0
    dfgpp["gpp_pred"] = gppmax*(1-np.exp(-cond_arr/gppmax*slope0))
#%% #%%
    dfgpp["gppR2"] = 1- np.mean((dfgpp.gpp-dfgpp.gpp_pred)**2)/np.var(dfgpp.gpp)

    #dfgpp["gppR2"] = np.corrcoef(dfgpp.gpp_pred, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_no_cond"] = np.corrcoef(dfgpp.gppmax, dfgpp.gpp)[0,1]**2
    dfgpp["gppR2_only_cond"] = np.corrcoef(dfgpp.cond, dfgpp.gpp)[0,1]**2
#%%
    cond_a3 = dfgpp.cond * np.sqrt(dfgpp.vpd) / np.sqrt(dfgpp.kgpp)
    dfgpp["bcond_lo_smc"] = np.nanmedian(cond_a3[lo_soil])
    dfgpp["bcond_hi_smc"] = np.nanmedian(cond_a3[hi_soil])
#%%s
    #dfgpp.kgpp = np.median(dfgpp.kgpp)
    # dfgpp["waterbal"] = 1*dfgpp.waterbal_x
    if len(dfgpp) < 10:
         continue
    #%%
    #dfgpp["waterbal"] = 1*dfgpp.sinterp_mean2
    if dfgpp.inflow.iloc[0] > 0:
        continue
    #dfgpp.kgpp *= dfgpp.gpp/dfgpp.gpp_pred
    #dfi = fit_tau_res(dfgpp.copy())#.copy()
    
    #dfgpp["waterbal"] = 1*dfgpp.sinterp_mean
    dfgpp = dfgpp.loc[np.isfinite(dfgpp.waterbal)].copy()
    #%%
    # z = np.array(dfgpp["waterbal"])
    # counter = 1
    # ans = np.zeros(len(z))
    # ans[0] = 1
    # for di in range(1,len(z),1):
    #     if counter <= 50:
    #         ans[di] = 1
    #     if z[di] < z[di-1] + 10:
    #         counter += 1
    #     else:
    #         counter = 0
    # #%%
    # dfgpp = dfgpp.loc[ans==1]
    
    #hwb = (np.min(dfgpp.waterbal) + np.max(dfgpp.waterbal))/2
    #dfgpp = dfgpp.loc[dfgpp.waterbal < hwb].copy()
    
    #doymid = (dfgpp.summer_end.iloc[0] + dfgpp.summer_start.iloc[0])/2
    #dfgpp = dfgpp.loc[dfgpp.doy < doymid]
    #dfgpp = dfgpp.loc[dfgpp.waterbal > -500]
    #dfgpp = dfgpp.loc[dfgpp.waterbal < 0]

    #%%
    if len(dfgpp) < 10:
         continue
#%%
    dfi = fit_tau_res(dfgpp.copy())
    #dfi = fit_tau_res_assume_max(dfgpp.copy(),5)
    #dfi = fit_tau_res_width(dfgpp.copy())

    dfi["max_limitation"] = np.min(dfi.et_tau/dfi.et_null)
    dfi["soil_max"] = np.max(dfi.waterbal)
    dfi["soil_min"] = np.min(dfi.waterbal)
    
#%%
    dfi["npoints"] = len(dfi)
    all_results.append(dfi)
#%%

#%%
all_results = pd.concat(all_results)
#%%

#%%
site_count = np.array(all_results.groupby("SITE_ID").count()["waterbal"])
site_year = np.array(all_results.groupby("SITE_ID").nunique()["year"])

#%%
df1 = all_results.groupby("SITE_ID").first().reset_index()
#df1["site_count"] = site_count
#df1["year_count"] = site_year

df1["Aridity"] = df1.mean_netrad / (df1.map_data / (18/1000 * 60*60*24) * 44200)
#df1["Aridity_gs"] = df1.gs_netrad / (df1.mgsp_data / (18/1000 * 60*60*24) * 44200)
#f_stat = (df1.etr2_smc-df1.etr2_null)/(1-df1.etr2_smc)*(df1.site_count-2)
#scipy.stats.f.cdf(f_stat,1,df1.site_count-2)
#%%
df_meta3 = df1.sort_values("gppR2")
df_meta3["gpp_rank"] = np.arange(len(df_meta3))
#%%
fig,ax = plt.subplots(1,1,figsize=(15,6))

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.gpp_rank,subI.gppR2,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xlabel("Rank",fontsize=24)
ax.set_ylabel(r"$R^2$ of GPP",fontsize=24)
#ax.axhline(0,color='k')

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )

#%%

df1 = df1.loc[df1.etr2_smc > 0]
#df1 = df1.loc[df1.gppR2 > 0]

df1 = df1.loc[df1.mat_data > 3]
#%%
#df1 = pd.merge(df1,bif_forest,on="SITE_ID",how="left")
#%%
def qt_gt1(x,q):
    return np.quantile(x[x >= 1],q)
def mean_gt1(x):
    return np.mean(x[x >= 1])
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
def get_lens2(x,c):
    x2 = 1*x
    bucket = 0
    new_list = []
    for j in range(len(x)):
        if x2[j] == 0:
            if bucket > 0:
                new_list.append(bucket)
                bucket = 0
            new_list.append(0)
        else:
            bucket += x2[j]
    if bucket > 0:
        new_list.append(bucket)
    x2 = np.array(new_list)
    x2[0] = c+1
    day_diff = np.diff(np.where(x2 > c)[0])
    return day_diff

#%%
rain_data = pd.read_csv("rain_50_50_mar1.csv")

rain_sites = pd.unique(df1.SITE_ID)
ddl_rain = []
ddl_rain2 = []
ddl_rain10 = []

rain_gs_mean = []
rain_pos_mean = []
for x in rain_sites:
    rain_site = rain_data.loc[rain_data.SITE_ID==x].copy()
    rain_allyear = np.array(rain_site.rain_mm)
    year_list = np.array(rain_site.year)
    site_rain_mean = np.mean(rain_allyear[np.isfinite(rain_allyear)])
    site_rain_pos = rain_allyear[np.isfinite(rain_allyear)*(rain_allyear > 0)]
    #site_rain_pos = site_rain_pos[site_rain_pos > np.quantile(site_rain_pos,0.25)]
    #site_rain_posmean = np.mean(rain_allyear[np.isfinite(rain_allyear)*(rain_allyear > 0)])

    rain_gs_mean.append(site_rain_mean)
    #rain_pos_mean.append(site_rain_posmean)

    years_max = []
    years_mean = []
    years_mean10 = []

    for y in np.unique(year_list):
        #cutoff = df_meta.map_data.loc[df_meta.SITE_ID==x].iloc[0]*4
        z = 1*rain_allyear[year_list==y]
        z[-1] = np.inf
       # years_max.append(np.max(get_lens(z,np.mean(site_rain_pos))))
        ly = get_lens(z,10)
        years_max.append(np.max(ly))
        years_mean.append(np.mean(ly[ly >= 2]))
        years_mean10.append(np.mean(ly[ly >= 10]))

        #years_max.append(np.max(interval_len(z,site_rain_mean/4)))


    ddl_rain.append(np.mean(years_max))
    ddl_rain2.append(np.mean(years_mean))
    ddl_rain10.append(np.mean(years_mean10))

#%%
rain_site_tab = pd.DataFrame({"SITE_ID":rain_sites,
                              "ddrain_mean":ddl_rain,
                              "gsrain_mean":rain_gs_mean,
                              "ddrain_2mean":ddl_rain2,
                              "ddrain_10mean":ddl_rain10})
df1 = pd.merge(df1,rain_site_tab,on="SITE_ID",how="left")

#%%

#%%
df_meta = df1.copy()

#%%
#df_meta["tau_rel_unc"] = (df_meta.tau_75-df_meta.tau_25)/df_meta.tau
#%%
#fval = ((1-df_meta.etr2_null)-(1-df_meta.etr2_smc))/(1-df_meta.etr2_smc)*(df_meta.npoints-4)
#df_meta["ftest"] = 1-scipy.stats.f.cdf(x=fval,dfn=1,dfd=df_meta.npoints-4)
#df_meta = df_meta.loc[ftest > 0.99]
#df_meta = df_meta.loc[df_meta.tau_rel_unc < 0.25].copy()
#%%
df_meta = df_meta.loc[df_meta.gppR2 > 0].copy()
#%%
#df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_no_cond > 0.05]
df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_only_cond > 0]
#%%
#df_meta = df_meta.loc[df_meta.smin > -10]
#%%
df_meta = df_meta.loc[df_meta.tau > 0]
#df_meta = df_meta.loc[df_meta.tau_lo > 0]
#df_meta = df_meta.loc[df_meta.tau_hi > 0]

#%%
df_meta["rel_err"] = (df_meta.etr2_smc-df_meta.etr2_null)#/(1-df_meta.etr2_null)
#%%
df_meta = df_meta.loc[df_meta.rel_err > 0.05]
#%%
#df_meta["rel_err_lim"] = (df_meta.etr2_limited_smc-df_meta.etr2_limited_null)/(1-df_meta.etr2_limited_null)

#%%
all_results["g_marginal_effect"] = 1-np.exp(-all_results.cond/all_results.kgpp)

resmean = all_results.groupby("SITE_ID").mean(numeric_only=True).reset_index()
df_meta = pd.merge(df_meta,resmean[["SITE_ID","kgpp","gpp","gpp_pred","cond","g_marginal_effect","waterbal"]],how='left',on='SITE_ID')
#df_meta = df_meta.loc[df_meta.summer_end - df_meta.summer_start - df_meta.tau > 0]
#%%
aunique = all_results.groupby("SITE_ID").nunique().reset_index().rename(columns={"year":"nyear"})
df_meta = pd.merge(df_meta,aunique[["SITE_ID","nyear"]],how="left",on="SITE_ID")
#df_meta = df_meta.loc[df_meta.nyear >= 3]
#%% 
amin = all_results.groupby("SITE_ID").min(numeric_only=True).reset_index().rename(columns={"waterbal":"wbal_min"})
df_meta = pd.merge(df_meta,amin[["SITE_ID","wbal_min"]],how="left",on="SITE_ID")
#%%
astd = all_results.groupby("SITE_ID").std(numeric_only=True).reset_index().rename(columns={"waterbal":"wbal_std"})
df_meta = pd.merge(df_meta,astd[["SITE_ID","wbal_std"]],how="left",on="SITE_ID")
#%%
df_meta["gsfrac"] = df_meta.npoints/(df_meta.summer_end-df_meta.summer_start)
#df_meta = df_meta.loc[df_meta.gsfrac > 1]
#%%
df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Fire"]
#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Agriculture"]
#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Grazing"]
#%%
#df_meta = df_meta.loc[df_meta.max_limitation < 0.9]
# df_meta["soil_rel_max"] = np.clip(df_meta.soil_max/1000 - df_meta.smin,0,100)
# df_meta["soil_rel_min"] = np.clip(df_meta.soil_min/1000 - df_meta.smin,0,100)
# df_meta["soil_ratio"] = df_meta["soil_rel_min"]/df_meta["soil_rel_max"]
# df_meta = df_meta.loc[np.sqrt(df_meta.soil_ratio) < 0.67]
#%%
#df_meta = df_meta.loc[df_meta.bcond_lo_smc/df_meta.bcond_hi_smc < 0.67]
#%%
rainmod = smf.ols("tau ~ ddrain_mean",data=df_meta).fit()
#rainmodW = smf.wls("tau ~ ddrain_mean",data=df_meta,weights=df_meta.etr2_smc).fit()
#rainmod_gpp_add = smf.ols("tau ~ ddrain_mean",data=df_meta).fit()
#rainmod_gpp2 = smf.ols("tau ~ ddrain_mean + kgpp_y",data=df_meta).fit()

#%%
fig,ax = plt.subplots(1,1,figsize=(10,8))

lmax = 500

line1, = ax.plot([0,lmax],[0,lmax],"k",label="1:1 line")
line2, = ax.plot([0,lmax],np.array([0,lmax])*rainmod.params[1]+rainmod.params[0],"b--",label="Regression line\n($R^2$ = 0.65)")
#plt.plot([0,150],np.array([0,150])*reg0.params[0],"b--",label="Regression line\n($R^2$ = 0.39)")
#leg1 = ax.legend(loc="upper left")
leg1 = ax.legend(loc="lower right")

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.ddrain_mean,subI.tau,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

ax.set_xlim(0,200)
ax.set_ylim(0,200)
ax.set_xlabel("Annual-mean $D_{max}$ (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")
#%%
fig,ax = plt.subplots(1,1,figsize=(10,8))

# lmax = 500

# line1, = ax.plot([0,lmax],[0,lmax],"k",label="1:1 line")
# line2, = ax.plot([0,lmax],np.array([0,lmax])*rainmod.params[1]+rainmod.params[0],"b--",label="Regression line\n($R^2$ = 0.65)")
# #plt.plot([0,150],np.array([0,150])*reg0.params[0],"b--",label="Regression line\n($R^2$ = 0.39)")
# #leg1 = ax.legend(loc="upper left")
# leg1 = ax.legend(loc="lower right")

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.ddrain_2mean,subI.tau,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,200)
ax.set_ylim(0,200)
ax.set_xlabel("Aridity index",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )



#%%
yscaler = np.sqrt(zsoil_mol)
molm2_to_mm = 18/1000
s2day = 60*60*24

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
ax.plot(df_meta.LOCATION_LONG,df_meta.LOCATION_LAT,'*',alpha=0.75,color="red",markersize=10,markeredgecolor="gray")
ax.set_xlim(np.min(df_meta.LOCATION_LONG)-7,np.max(df_meta.LOCATION_LONG)+7)
ax.set_ylim(np.min(df_meta.LOCATION_LAT)-7,np.max(df_meta.LOCATION_LAT)+7)
#%%
df_meta2 = df1.copy()

df_meta2["combined_biome"] = [simple_biomes[x] for x in df_meta2["IGBP"]]

df_meta2 = df_meta2.loc[df_meta2.gppR2 > 0.0].copy()

df_meta2 = df_meta2.loc[df_meta2.gppR2-df_meta2.gppR2_only_cond > 0.0]

#df_meta2 = df_meta2.loc[df_meta2.tau > 0]
#df_meta = df_meta.loc[df_meta.tau_lo > 0]
#df_meta = df_meta.loc[df_meta.tau_hi > 0]
df_meta2 = df_meta2.loc[df_meta2.DOM_DIST_MGMT != "Fire"]
#df_meta2 = df_meta2.loc[df_meta2.DOM_DIST_MGMT != "Agriculture"]
#df_meta2 = df_meta2.loc[df_meta2.DOM_DIST_MGMT != "Grazing"]
#%%

fig,ax = plt.subplots(1,1,figsize=(10,8))

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.mat_data,subI.map_data*365/10,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,210)
#ax.set_ylim(0,210)
ax.set_xlabel("Average temperature ($^oC$",fontsize=24)
ax.set_ylabel("Average annual precip. (cm)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%
xarr = np.linspace(0,900,500)

site_id = "US-Me4"
df2 = all_results.loc[all_results.SITE_ID==site_id].copy()
#df2.cond = df2.ET / df2.vpd * 100
#df2 = fit_tau_res_assume_max(df2.copy(),2)

tauI = df2.tau.iloc[0]*60*60*24
#tauS = tauI*60

plt.figure()
var_combo = 2*zsoil_mol*(df2.waterbal/1000-1*df2.smin)/(df2.vpd/100)*df2.kgpp

plt.plot(np.sqrt(var_combo),df2.cond-df2.res_cond,'ro',alpha=0.5,label=r"US-Me4, $\tau$ = 170 days")
print(df2.tau.iloc[0])
print(df2.smin.iloc[0])
gmax_i = df2.gmax.iloc[0]
plt.plot(xarr,np.clip(xarr/np.sqrt(tauI),0,gmax_i),'r',linewidth=3)

site_id = "DE-Obe"
df2 = all_results.loc[all_results.SITE_ID==site_id].copy()#.iloc[300:]

#df2.cond = df2.ET / df2.vpd * 100
#df2 = fit_tau_res_assume_max(df2.copy(),2)

tauI = df2.tau.iloc[0]*60*60*24
#tauI = 40*60*60*24
var_combo = 2*zsoil_mol*(df2.waterbal/1000-1*df2.smin)/(df2.vpd/100)*df2.kgpp
plt.plot(np.sqrt(var_combo),df2.cond-df2.res_cond,'bo',alpha=0.5,label=r"DE-Obe, $\tau$ = 23 days")
print(df2.tau.iloc[0])
print(df2.smin.iloc[0])
gmax_i = df2.gmax.iloc[0]
plt.plot(xarr,np.clip(xarr/np.sqrt(tauI),0,gmax_i),'b',linewidth=3)

#plt.xlim(0,1e6)
plt.ylim(0,0.65)

plt.legend(framealpha=1)

plt.xlabel(r"$(2g_A (S-S_0)/VPD)^{0.5}$ $(mol/m^2/s)$",fontsize=24)
plt.ylabel("g from eddy covariance\n$(mol/m^2/s)$",fontsize=24)
#%%

#%%
df_meta3 = df_meta.sort_values("etr2_smc")
df_meta3["et_rank"] = np.arange(len(df_meta3))
#%%
fig,ax = plt.subplots(1,1,figsize=(15,6))

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.et_rank,subI.etr2_smc,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xlabel("Rank",fontsize=24)
ax.set_ylabel(r"$R^2$ of ET",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%
df_meta3 = df_meta.sort_values("gr2_smc")
df_meta3["g_rank"] = np.arange(len(df_meta3))
#%%
fig,ax = plt.subplots(1,1,figsize=(15,6))

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.g_rank,subI.gr2_smc,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xlabel("Rank",fontsize=24)
ax.set_ylabel(r"$R^2$ of g",fontsize=24)
#ax.axhline(0,color='k')

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%
plt.figure(figsize= (10,10))
plt.subplot(2,1,1)
plt.plot(all_results.drel_both, np.log(all_results.gpp/all_results.gpp_pred),'.',alpha=0.1); 
plt.ylim(-1,1);
plt.ylabel("log GPP error")
plt.subplot(2,1,2)
plt.plot(all_results.drel_both, np.log(all_results.ET/all_results.et_tau),'.',alpha=0.1); 
plt.ylim(-1,1);
plt.ylabel("log ET error")

plt.xlabel("Day relative to growing season peak")
#%%
plt.figure()
plt.plot(np.log(all_results.gpp/all_results.gpp_pred), np.log(all_results.ET/all_results.et_tau),'.',alpha=0.1); 
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.xlabel("log GPP error")
plt.ylabel("log ET error")

#%%

#plt.figure()
#plt.plot(all_results.gpp-all_results.gpp_pred, all_results.ET-all_results.et_tau,'.',alpha=0.1); 
#%%
# plt.ylim(-1,1)
# plt.xlim(-1,1)
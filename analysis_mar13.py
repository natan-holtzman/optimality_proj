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
from fit_tau_res_cond2 import fit_tau_res, fit_tau_res_evi , fit_tau_res_assume_max, fit_tau_res_assume_max_smin, fit_tau_res_width
from gpp_funs_mar13 import fit_gpp,fit_gpp3, fit_gpp_tonly, fit_gpp_nopar
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
dry_list = pd.read_csv("dry_site_list.csv")
bif_forest["is_dry_limited"] = bif_forest.SITE_ID.isin(dry_list.SITE_ID)
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
#df_in = pd.read_csv("gs50_constGS_evi_lai.csv")
#df_in = pd.read_csv("gs_50_laiGS_mar12b.csv")

#df2 = pd.read_csv("gs50_mar13_lai_evi.csv")
df_in = pd.read_csv("gs_67_laiGS_mar16.csv")
#df_in = pd.read_csv("gs_50_laiGS_mar14_nosat.csv")
#df_in = pd.read_csv("gs_50_laiGS_mar16.csv")
#df_in = pd.merge(df_in,df2[["SITE_ID","EVIint","EVIside","date"]],on=["SITE_ID","date"],how='left')

rain_data = pd.read_csv("rain_67_mar16.csv")

#%%
#df_in = pd.read_csv("gs50_varGS_evi_lai.csv")
#df_in = pd.merge(df_in,df2[["SITE_ID","date","summer_peak","summer_start","summer_end"]],on=["SITE_ID","date"],how='left')
#df2 = None
#%%
df_in = pd.merge(df_in,bif_forest,on="SITE_ID",how='left')
#%%
df_in["res_cond"] = 0


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

df_in["combined_biome"] = [simple_biomes[x] for x in df_in["IGBP"]]

#%%

df_in = df_in.loc[df_in["gpp"]  > 0]
#df_in = df_in.loc[df_in["par"]  > 150]
#%%
df_in["drel_spring"] = -np.clip(df_in["doy"] - df_in["summer_peak"],-np.inf,0) / (df_in["summer_peak"] - df_in["summer_start"])
df_in["drel_fall"] = np.clip(df_in["doy"] - df_in["summer_peak"],0,np.inf) / (df_in["summer_end"] - df_in["summer_peak"])
df_in["drel_both"] = -df_in["drel_spring"] + df_in["drel_fall"]
#%%
df_in = df_in.loc[np.isfinite(df_in.drel_both)]
#%%
#dfgpp_together = fit_gpp_flex_slope_TP_nores_laiday(df_in)
daytab = pd.read_csv("site_daylight.csv",parse_dates=["date"])
daytabH = pd.read_csv("site_hourly_daylight.csv",parse_dates=["date"])
daytab = pd.concat([daytab,daytabH]).reset_index()

daytab["doy_raw"] = daytab.date.dt.dayofyear
#%%
daytab_avg = daytab.groupby(["SITE_ID","doy_raw"]).mean(numeric_only=True).reset_index()
#%%
df_in = pd.merge(df_in,daytab_avg[["SITE_ID","doy_raw","SW_IN_POT","NIGHT"]],on=["SITE_ID","doy_raw"],how="left")
df_in = df_in.loc[np.isfinite(df_in.NIGHT)]
#%%site
all_results = []

for site_id in pd.unique(df_in.SITE_ID)[:]:#[forest_daily[x] for x in [70,76]]:
#%%
    print(site_id)
    dfgpp = df_in.loc[df_in.SITE_ID==site_id].copy()
    dfgpp = dfgpp.loc[dfgpp.airt > 5]
    #dfgpp = dfgpp.loc[dfgpp.EVIside >= 0]
    #dfgpp = dfgpp.loc[dfgpp.drel_both >= 0]
    #%%
    
    
    #dfgpp = dfgpp.loc[np.isfinite(dfgpp.EVIint)]
    #dfgpp = dfgpp.loc[np.isfinite(dfgpp.EVI2)]

    #%%
    if len(dfgpp) == 0:
        continue
    #%%
    #dfgpp["dayfrac"] = 1-dfgpp.NIGHT
    #%%
    y1 = np.array(1-dfgpp.NIGHT)
    x1 = np.array(dfgpp.doy*2*np.pi/365)
    xstack = sm.add_constant(np.stack((np.cos(x1),np.sin(x1)),1))
    r1 = sm.OLS(y1,xstack).fit()
    dfgpp["dayfrac"] = r1.fittedvalues
    #%%
    #dfgpp.LAI = (dfgpp.EVIint - np.mean(dfgpp.EVIint))/np.std(dfgpp.EVIint)*np.std(dfgpp.LAI) + np.mean(dfgpp.LAI)
#%%
    cn = 1*dfgpp.cond
    cn -= np.mean(cn)
    dfgpp = dfgpp.loc[np.abs(cn) < 3*np.std(cn)]
    
    lcn = np.log(dfgpp.cond)
    lcn -= np.mean(lcn)
    dfgpp = dfgpp.loc[np.abs(lcn) < 3*np.std(lcn)]
    #%%
    
    dfgpp.dayfrac /= np.max(dfgpp.dayfrac)
    
    #dfgpp.LAI = 1*dfgpp.EVIint
    #dfgpp.LAI = 1.0
    dfgpp.LAI /= np.max(dfgpp.LAI)
    #dfgpp = dfgpp.loc[dfgpp.LAI > 0.9]
    #dfgpp.LAI = 1.0

#    dfgpp = dfgpp.loc[dfgpp.dayfrac > 0.9]
    
    #%%
    # dfgpp["normcond"] = dfgpp.cond/dfgpp.LAI/dfgpp.dayfrac
    # dfgpp["normgpp"] = dfgpp.gpp/dfgpp.LAI/dfgpp.dayfrac
    # dfgpp["normrad"] = dfgpp.par/dfgpp.dayfrac
    # dfgpp = fit_gpp(dfgpp.copy())
    #%%
    #dfgpp.LAI = 1*dfgpp.EVI2
    #dfgpp.LAI /= np.max(dfgpp.LAI)
    dfgpp["normcond"] = dfgpp.cond/dfgpp.LAI/dfgpp.dayfrac
    dfgpp["normgpp"] = dfgpp.gpp/dfgpp.LAI/dfgpp.dayfrac
    dfgpp["normrad"] = dfgpp.par/dfgpp.dayfrac
    dfgpp = fit_gpp3(dfgpp.copy())
    #%%
    # dfgpp.LAI = 1*dfgpp.EVI2
    # dfgpp.LAI /= np.max(dfgpp.LAI)
    # dfgpp["normcond"] = dfgpp.cond/dfgpp.LAI/dfgpp.dayfrac
    # dfgpp["normgpp"] = dfgpp.gpp/dfgpp.LAI/dfgpp.dayfrac
    # dfgpp["normrad"] = dfgpp.par/dfgpp.dayfrac
    # dfgpp2 = fit_gpp_tonly(dfgpp.copy())
#%%
    # if dfgpp1.gppR2.iloc[0] > dfgpp2.gppR2.iloc[0]:
    #     dfgpp = dfgpp1.copy()
    # else:
    #     dfgpp = dfgpp2.copy()
    #%%
    lo_soil = dfgpp.waterbal < np.nanquantile(dfgpp.waterbal,0.25)
    hi_soil = dfgpp.waterbal > np.nanquantile(dfgpp.waterbal,0.75)
    
    #lo_soil = dfgpp.smc < np.nanquantile(dfgpp.smc,0.25)
    #hi_soil = dfgpp.smc > np.nanquantile(dfgpp.smc,0.75)
    dfgpp["cond_lo_smc"] = np.nanmedian(dfgpp.cond[lo_soil])
    dfgpp["cond_hi_smc"] = np.nanmedian(dfgpp.cond[hi_soil])

    cond_dq = dfgpp.cond * np.sqrt(dfgpp.vpd)
    dfgpp["acond_lo_smc"] = np.nanmedian(cond_dq[lo_soil])
    dfgpp["acond_hi_smc"] = np.nanmedian(cond_dq[hi_soil])
    
    cond_dq = dfgpp.normcond
    dfgpp["ncond_lo_smc"] = np.nanmedian(cond_dq[lo_soil])
    dfgpp["ncond_hi_smc"] = np.nanmedian(cond_dq[hi_soil])
    
    cond_dq = dfgpp.normcond * np.sqrt(dfgpp.vpd)
    dfgpp["ancond_lo_smc"] = np.nanmedian(cond_dq[lo_soil])
    dfgpp["ancond_hi_smc"] = np.nanmedian(cond_dq[hi_soil])
    
    cond_dq = dfgpp.ET / dfgpp.LAI/dfgpp.dayfrac
    dfgpp["nE_lo_smc"] = np.nanmedian(cond_dq[lo_soil])
    dfgpp["nE_hi_smc"] = np.nanmedian(cond_dq[hi_soil])
    
    #%%
    if len(dfgpp) < 10:
        continue

    #%%
    if np.min(dfgpp.gppmax) < 0:
        continue
    if np.max(dfgpp.gppmax) > 100:
        continue
    if np.min(dfgpp.kgpp) <= 0:
         continue
    #%%
    lo_soil = dfgpp.waterbal < np.nanquantile(dfgpp.waterbal,0.25)
    hi_soil = dfgpp.waterbal > np.nanquantile(dfgpp.waterbal,0.75)
    cond_a3 = dfgpp.cond * np.sqrt(dfgpp.vpd) / np.sqrt(dfgpp.kgpp)
    dfgpp["bcond_lo_smc"] = np.nanmedian(cond_a3[lo_soil])
    dfgpp["bcond_hi_smc"] = np.nanmedian(cond_a3[hi_soil])
#%%s
    #dfgpp.kgpp = np.median(dfgpp.kgpp)
    # dfgpp["waterbal"] = 1*dfgpp.waterbal_x
    if len(dfgpp) < 10:
         continue
    #%%
    #dfgpp["waterbal"] = 1*dfgpp.sinterp_anom
    dfgpp = dfgpp.loc[np.isfinite(dfgpp.waterbal)].copy()
    #%%
    if len(dfgpp) < 10:
        continue
    #%%
    try:
        dfgpp["sgcor"] = cor_skipna(dfgpp.smc,dfgpp.cond)[0]
    except:
        dfgpp["sgcor"] = np.nan
    #%%
    if dfgpp.inflow.iloc[0] > 0:
        continue
    #dfgpp.kgpp *= dfgpp.gpp/dfgpp.gpp_pred
    #dfi = fit_tau_res(dfgpp.copy())#.copy()
    
    #dfgpp["waterbal"] = 1*dfgpp.sinterp_mean
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
    # dfgpp = dfgpp.loc[ans==1]

    #%%
    if len(dfgpp) < 10:
         continue
#%%
    dfgpp["res_cond"] = 0.0
    #dfi = fit_tau_res(dfgpp.copy())
    dfi = fit_tau_res_assume_max(dfgpp.copy(),10)
    #dfi = fit_tau_res_width(dfgpp.copy()

    dfi["max_limitation"] = np.min(dfi.et_tau/dfi.et_null)
    dfi["soil_max"] = np.max(dfi.waterbal)
    dfi["soil_min"] = np.min(dfi.waterbal)
    #%%
    #dfi["ga2"] = dfi.g_adj**2
    #dfi["wb2"] = dfi.waterbal/1000
    #r1 = smf.wls("ga2 ~ wb2",data=dfi,weights=1/dfi.g_adj).fit()
    #%%
    
    nbs = 25
    bstau = np.zeros(nbs)
   # dflist = []
    #bsmin = np.zeros(nbs)
    for bsi in range(nbs):
        # bdf = dfgpp.sample(len(dfgpp),replace=True)
        # df2 = fit_gpp_nopar(bdf)
        # df2 = df2.loc[df2.kgpp > 0]
        df2 = dfgpp.sample(len(dfgpp),replace=True)
        df3 = fit_tau_res_assume_max(df2,10)
        bstau[bsi] = df3.tau.iloc[0]
       # dflist.append(df3)
        #bsmin[bsi] = df3.smin.iloc[0]
        
    dfi["tau_bs_std"] = np.std(bstau)
#%%
    #r2 = scipy.stats.theilslopes(dfi.ga2,dfi.wb2)
    #%%
    # smin_range = np.linspace(0.75,1.25,50)*dfi.smin.iloc[0]
    # myslopes = []
    # myse = []
    # myr2 = []
    # for x in smin_range:
    #     regi = sm.OLS(dfi.g_adj, np.sqrt(np.clip(dfi.wb2-x,0,np.inf))).fit()
    #     myslopes.append(regi.params[0])
    #     myse.append(regi.bse[0])
    #     myr2.append(regi.rsquared)
    #%%
    #plt.plot(dfi.waterbal,dfi.ET/dfi.et_null,'.'); plt.plot(dfi.waterbal,dfi.et_tau/dfi.et_null,'.');
    #%%
    #plt.plot(dfi.waterbal/1000,dfi.g_adj,'.')
    #xarr = np.linspace(-0.6,0,100)
    # plt.plot(xarr,np.sqrt((xarr-dfi.smin.iloc[0])/(dfi.tau.iloc[0]*60*60*24)),linewidth=3)
    # i0 = -r1.params[0]/r1.params[1]
    # tau0 = 1/(r1.params[1]*60*60*24)
    # plt.plot(xarr,np.sqrt((xarr-i0)/(tau0*60*60*24)),linewidth=3)
    
    # i1 = -r2.intercept/r2.slope
    # tau1 = 1/(r2.slope*60*60*24)
    # dfi["tauTS"] = tau1
    # dfi["sminTS"] = i1
    # dfi["tau_hi"] = 1/(r2.low_slope*60*60*24)
    # dfi["tau_lo"] = 1/(r2.high_slope*60*60*24)

    #plt.plot(xarr,np.sqrt((xarr-i1)/(tau1*60*60*24)),linewidth=3)
    
    # dfgpp_constVPD = dfgpp.copy()
    # dfgpp_constVPD.vpd = np.mean(dfgpp_constVPD.vpd)
    # dfi_constVPD = fit_tau_res_assume_max(dfgpp_constVPD,10)
    # #%%
    # dfgpp_constK = dfgpp.copy()
    # dfgpp_constK.kgpp = np.mean(dfgpp_constK.kgpp)
    # dfi_constK = fit_tau_res_assume_max(dfgpp_constK,10)
    #%%
    seas_plot = 0
    if seas_plot:
        #%%
        plt.figure(figsize= (10,10))
        plt.subplot(2,1,1)
        plt.plot(dfi.doy, dfi.gpp-dfi.gpp_pred,'o',alpha=0.67); 
        plt.ylabel("GPP residual")
        plt.subplot(2,1,2)
        plt.plot(dfi.doy, (dfi.ET-dfi.et_tau)*18/1000*24*60*60,'o',alpha=0.67); 
        plt.ylabel("ET residual")
        plt.xlabel("Day of year")
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
df1["Aridity_gs"] = df1.gs_netrad / (df1.mgsp_data / (18/1000 * 60*60*24) * 44200)
#f_stat = (df1.etr2_smc-df1.etr2_null)/(1-df1.etr2_smc)*(df1.site_count-2)
#scipy.stats.f.cdf(f_stat,1,df1.site_count-2)
#%%
df_tocompare = df1.copy()
#%%
#df_tocompare = df_tocompare.loc[df_tocompare.gppR2-df_tocompare.gppR2_no_cond2 > 0.01]
#df_tocompare = df_tocompare.loc[df_tocompare.gppR2-df_tocompare.gppR2_only_cond > 0.01]
#%%
#%%

df1 = df1.loc[df1.etr2_smc > 0]
#df1 = df1.loc[df1.gppR2 > 0]

#df1 = df1.loc[df1.mat_data > 3]
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
#rain_data = pd.read_csv("rain_all_mar2.csv")

rain_sites = pd.unique(df1.SITE_ID)
ddl_rain = []
ddl_rain2 = []
ddl_rain10 = []
gsrain = []
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
    years_gslen = []

    for y in np.unique(year_list):
        #cutoff = df_meta.map_data.loc[df_meta.SITE_ID==x].iloc[0]*4
        z = 1*rain_allyear[year_list==y]
        z[-1] = np.inf
        z[0] = np.inf

       # years_max.append(np.max(get_lens(z,np.mean(site_rain_pos))))
        ly = get_lens(z,5)
        years_max.append(np.max(ly))
        years_mean.append(np.mean(ly[ly >= 2]))
        years_mean10.append(np.mean(ly[ly >= 10]))
        years_gslen.append(len(z))

        #years_max.append(np.max(interval_len(z,site_rain_mean/4)))


    ddl_rain.append(np.mean(years_max))
    ddl_rain2.append(np.mean(years_mean))
    ddl_rain10.append(np.mean(years_mean10))
    gsrain.append(np.mean(years_gslen))

#%%
rain_site_tab = pd.DataFrame({"SITE_ID":rain_sites,
                              "ddrain_mean":ddl_rain,
                              "gsrain_mean":rain_gs_mean,
                              "ddrain_2mean":ddl_rain2,
                              "ddrain_10mean":ddl_rain10,
                              "gsrain_len":gsrain})
df1 = pd.merge(df1,rain_site_tab,on="SITE_ID",how="left")


#%%
df_meta = df1.copy()

#fval = ((1-df_meta.etr2_null)-(1-df_meta.etr2_smc))/(1-df_meta.etr2_smc)*(df_meta.npoints-4)
#df_meta["ftest"] = 1-scipy.stats.f.cdf(x=fval,dfn=1,dfd=df_meta.npoints-4)
#df_meta = df_meta.loc[df_meta.ftest < 0.01]
#df_meta = df_meta.loc[df_meta.LOCATION_LAT > 0]
#df_meta = df_meta.loc[df_meta.tau_rel_unc < 0.25].copy()

df_meta = df_meta.loc[df_meta.gppR2 > 0.01].copy()

#df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_no_cond > 0.01]
#df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_only_cond > 0.01]

#df_meta = df_meta.loc[df_meta.sgcor < 0.67]

df_meta = df_meta.loc[df_meta.tau > 0]
#df_meta = df_meta.loc[df_meta.tau_lo > 0]
#df_meta = df_meta.loc[df_meta.tau_hi > 0]


df_meta["rel_err"] = (df_meta.etr2_smc-df_meta.etr2_null)#/(1-df_meta.etr2_null)
#df_meta = df_meta.loc[df_meta.rel_err > 0.05]
#df_meta = df_meta.loc[df_meta.ancond_lo_smc/df_meta.ancond_hi_smc < 0.75]
#df_meta = df_meta.loc[df_meta.bcond_lo_smc/df_meta.bcond_hi_smc < 0.67]

#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Fire"]
#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Agriculture"]
#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Grazing"]

#df_meta = df_meta.loc[df_meta.nE_lo_smc/df_meta.nE_hi_smc < 0.8]
#df_meta = df_meta.loc[df_meta.max_limitation < 0.9]
df_meta["soil_rel_max"] = np.clip(df_meta.soil_max/1000 - df_meta.smin,0,100)
df_meta["soil_rel_min"] = np.clip(df_meta.soil_min/1000 - df_meta.smin,0,100)
df_meta["soil_ratio"] = df_meta["soil_rel_min"]/df_meta["soil_rel_max"]
#%%
df_meta = df_meta.loc[df_meta.tau_bs_std < 30]
# df_meta = df_meta.loc[df_meta.tauTS > 0]
# df_meta = df_meta.loc[df_meta.tau_hi > 0]
# df_meta = df_meta.loc[df_meta.tau_lo > 0]
# df_meta = df_meta.loc[df_meta.tau_hi/df_meta.tau_lo < 2]
#%%
#df_meta = df_meta.loc[np.sqrt(df_meta.soil_ratio) < 0.75]
#%%
#df_meta = df_meta.loc[df_meta.ancond_lo_smc/df_meta.ancond_hi_smc < 0.75]
#%%s
#df_meta = df_meta.loc[df_meta.bcond_lo_smc/df_meta.bcond_hi_smc < 0.75]
#%%
rainmod = smf.ols("tau ~ ddrain_mean",data=df_meta).fit()
#%%
r2_11 = 1-np.mean((df_meta.ddrain_mean-df_meta.tau)**2)/np.var(df_meta.tau)
print(r2_11)

fig,ax = plt.subplots(1,1,figsize=(10,8))

lmax = 1.1*np.max(df_meta.ddrain_mean)

#line1, = ax.plot([0,lmax],[0,lmax],"k",label="1:1 line, $R^2$=0.59")
betas = np.array(np.round(np.abs(rainmod.params),2)).astype(str)
if rainmod.params[0] < 0:
    reg_eqn = r"$\tau$ = "+betas[1]+"$D_{max}$"+" - "+betas[0]
else:
    reg_eqn = r"$\tau$ = "+betas[1]+"$D_{max}$"+" + "+betas[0]
r2_txt = "($R^2$ = " + str(np.round(rainmod.rsquared,2)) + ")"
reg_lab = "Regression line" + "\n" + reg_eqn + "\n" + r2_txt
line2, = ax.plot([0,lmax],np.array([0,lmax])*rainmod.params[1]+rainmod.params[0],"b--",label=reg_lab)
#plt.plot([0,150],np.array([0,150])*reg0.params[0],"b--",label="Regression line\n($R^2$ = 0.39)")
#leg1 = ax.legend(loc="upper left")
#leg1 = ax.legend(loc="lower right")
leg1 = ax.legend()

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.ddrain_mean,subI.tau,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
xmax = np.max(df_meta.ddrain_mean)
ymax = np.max(df_meta.tau)


ax.set_xlim(0,1.1*xmax)
ax.set_ylim(0,1.1*ymax)
ax.set_xlabel("Annual-mean $D_{max}$ (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")
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

site_id = "US-Blo"
df2 = all_results.loc[all_results.SITE_ID==site_id].copy()
#df2.cond = df2.ET / df2.vpd * 100
#df2 = fit_tau_res_assume_max(df2.copy(),2)

tauI = df2.tau.iloc[0]*60*60*24
lab1 = site_id + r", $\tau$ = " + str(int(np.round(df2.tau.iloc[0],0))) + " days"
#tauS = tauI*60

plt.figure()
var_combo = 2*zsoil_mol*(df2.waterbal/1000-1*df2.smin)/(df2.vpd/100)*df2.kgpp

plt.plot(np.sqrt(var_combo),df2.cond-1*df2.res_cond,'ro',alpha=0.5,label=lab1)
print(df2.tau.iloc[0])
print(df2.smin.iloc[0])
gmax_i = df2.gmax.iloc[0]
#plt.plot(xarr,np.clip(xarr/np.sqrt(tauI),0,gmax_i),'r',linewidth=3)
plt.plot(xarr,np.clip(xarr/np.sqrt(tauI),0,np.inf),'k',linewidth=3)


site_id = "DE-Obe"
df2 = all_results.loc[all_results.SITE_ID==site_id].copy()#.iloc[300:]

#df2.cond = df2.ET / df2.vpd * 100
#df2 = fit_tau_res_assume_max(df2.copy(),2)

tauI = df2.tau.iloc[0]*60*60*24
lab2 = site_id + r", $\tau$ = " + str(int(np.round(df2.tau.iloc[0],0))) + " days"

#tauI = 40*60*60*24
var_combo = 2*zsoil_mol*(df2.waterbal/1000-1*df2.smin)/(df2.vpd/100)*df2.kgpp
plt.plot(np.sqrt(var_combo),df2.cond-1*df2.res_cond,'bo',alpha=0.5,label=lab2)
print(df2.tau.iloc[0])
print(df2.smin.iloc[0])
gmax_i = df2.gmax.iloc[0]
#plt.plot(xarr,np.clip(xarr/np.sqrt(tauI),0,gmax_i),'b',linewidth=3)
plt.plot(xarr,np.clip(xarr/np.sqrt(tauI),0,np.inf),'k',linewidth=3)

plt.xlim(0,1200)
plt.ylim(0,0.7)

plt.legend(framealpha=1)

plt.xlabel(r"$(2g_A (S-S_0)/VPD)^{0.5}$ $(mol/m^2/s)$",fontsize=24)
plt.ylabel("g from eddy covariance\n$(mol/m^2/s)$",fontsize=24)
#%%

#%%
df_meta3 = df_meta.sort_values("etr2_smc")
df_meta3["et_rank"] = np.arange(len(df_meta3))

fig,axes = plt.subplots(3,1,figsize=(15,10))
ax = axes[0]

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.et_rank,subI.etr2_smc,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
ax.set_xticks(df_meta3.et_rank,df_meta3.SITE_ID,rotation=90)
#ax.set_xlim(0,250)
ax.set_ylim(0,1)
#ax.set_xlabel("Rank",fontsize=24)
ax.set_ylabel(r"$R^2$ of ET",fontsize=24)

#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

df_meta3 = df_meta.sort_values("gr2_smc")
df_meta3["g_rank"] = np.arange(len(df_meta3))
ax = axes[1]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.g_rank,subI.gr2_smc,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.g_rank,df_meta3.SITE_ID,rotation=90)
ax.set_ylabel(r"$R^2$ of g",fontsize=24)
#ax.axhline(0,color='k')


df_meta3 = df_meta.sort_values("gppR2")
df_meta3["gpp_rank"] = np.arange(len(df_meta3))
ax = axes[2]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.gpp_rank,subI.gppR2,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.gpp_rank,df_meta3.SITE_ID,rotation=90)
ax.set_ylabel(r"$R^2$ of GPP",fontsize=24)
fig.tight_layout()
fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.02),ncols=2)
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%
plt.figure(figsize= (10,10))
plt.subplot(2,1,1)
plt.plot(all_results.drel_both, np.log(all_results.gpp/all_results.gpp_pred),'.',alpha=0.1); 
plt.ylim(-1,1);
plt.xlim(-2,2);

plt.ylabel("log GPP error")
plt.subplot(2,1,2)
plt.plot(all_results.drel_both, np.log(all_results.ET/all_results.et_tau),'.',alpha=0.1); 
plt.ylim(-1,1);
plt.xlim(-2,2);
plt.ylabel("log ET error")

plt.xlabel("Day relative to growing season peak")
#%%
lerr_gpp = np.log(all_results.gpp/all_results.gpp_pred)
lerr_et = np.log(all_results.ET/all_results.et_tau)
ok_err = (np.abs(lerr_gpp) < 0.5)*(np.abs(lerr_et) < 0.5)
print(cor_skipna(lerr_gpp[ok_err],lerr_et[ok_err]))
#%%
plt.figure()
plt.plot(lerr_gpp, lerr_et,'.',alpha=0.1); 
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
#%%
biome_index = dict(zip(biome_list,range(len(biome_list))))
df_meta["biome_number"] = [biome_index[x] for x in df_meta.combined_biome]
#%%
plot_colors = mpl.colormaps["tab10"](df_meta["biome_number"] +2)

#%%
fig,axes=plt.subplots(3,2,figsize=(8,10))
ax = axes[0,0]
ax.scatter(df_meta.Aridity,df_meta.tau,c=plot_colors)
ax.set_xlabel("Annual aridity index")
ax.set_ylabel(r"$\tau$ (days)")

ax = axes[0,1]
ax.scatter(df_meta.Aridity_gs,df_meta.tau,c=plot_colors)
ax.set_xlabel("GS aridity index")

ax = axes[1,0]
ax.scatter(df_meta.map_data,df_meta.tau,c=plot_colors)
ax.set_xlabel("Annual P (mm/day)")
ax.set_ylabel(r"$\tau$ (days)")

ax = axes[1,1]
ax.scatter(df_meta.gsrain_mean,df_meta.tau,c=plot_colors)
ax.set_xlabel("GS P (mm/day)")

ax = axes[2,0]
ax.scatter(df_meta.ddrain_2mean,df_meta.tau,c=plot_colors)
ax.set_xlabel("$D_{mean}$ (days)")
ax.set_ylabel(r"$\tau$ (days)")


ax = axes[2,1]
#ax.scatter(df_meta.summer_end-df_meta.summer_start,df_meta.tau,c=plot_colors)
ax.scatter(df_meta.gsrain_len,df_meta.tau,c=plot_colors)

ax.set_xlabel("GS length (days)")

fig.tight_layout()

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#%%
plt.figure(figsize=(7,7))
#plt.plot(df_meta.summer_end-df_meta.summer_start,df_meta.ddrain_mean,'o')
plt.plot(df_meta.gsrain_len,df_meta.ddrain_mean,'o')

plt.xlabel("GS length (days)",fontsize=22)
plt.ylabel("$D_{max}$ (days)",fontsize=22)
#%%
#df_meta = df_meta.loc[df_meta.summer_end-df_meta.summer_start < 300]
#%%
#plt.plot(df_meta.summer_end-df_meta.summer_peak, df_meta.tau,'o')

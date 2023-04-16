# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:46:32 2022

@author: nholtzma
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
#See what happens to ET r2 if set tau to est from regression
#and if set it to all-site mean

#%%
#from fit_tau_res_cond2 import fit_tau_res, fit_tau_res_evi , fit_tau_res_assume_max, fit_tau_res_assume_max_vpd, fit_tau_res_assume_max_smin, fit_tau_res_width, fit_assume_tau_res_max, fit_tau_year_effect
#from gpp_funs_mar13 import fit_gpp,fit_gpp3, fit_gpp_tonly, fit_gpp_nopar
from gpp_funs_mar19 import fit_gpp_lai_only2, fit_gpp_linear2
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

def r2_skipna(pred,obs):
    goodxy = np.isfinite(pred*obs)
    return 1 - np.mean((pred-obs)[goodxy]**2)/np.var(obs[goodxy])
#%%
bif_data = pd.read_csv("fn2015_bif_tab_h.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])].copy()
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH","CRO"])].copy()
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
#zsoil_mm_base = 1000
#zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
#gammaV = 0.07149361181458612
#width= np.inf
mol_s_to_mm_day = 1*18/1000*24*60*60

#gmax = np.inf
#%%

rain_dict = {}
year_tau_dict = {}
site_result = {}
#%%
# #%%
#df_in = pd.read_csv("gs_67_laiGS_mar16.csv")
#df_evi = pd.read_csv("gs_50_laiGS_mar16.csv")[["SITE_ID","EVI2","date"]]
# df_in = pd.merge(df_in,df_evi,on=["SITE_ID","date"],how='left')

#rain_data = pd.read_csv("rain_67_mar16.csv")
#%%
df_base = pd.read_csv("gs_67_mar23_evi.csv").groupby("SITE_ID").first().reset_index()

df_base["Aridity"] = df_base.mean_netrad / (df_base.map_data / (18/1000 * 60*60*24) * 44200)
df_base["Aridity_gs"] = df_base.gs_netrad / (df_base.mgsp_data / (18/1000 * 60*60*24) * 44200)

#%%
#df_in = pd.read_csv("gs_67_april3b_complete.csv")#[["SITE_ID","EVI2","date"]]
#rain_data = pd.read_csv("rain_67_mar31.csv")

#df_in = pd.read_csv("gs_80max_april10.csv")#[["SITE_ID","EVI2","date"]]
#rain_data = pd.read_csv("rain_80max_april10.csv")

#df_in = pd.read_csv("gs_67max_gpp_april7.csv")#[["SITE_ID","EVI2","date"]]
#rain_data = pd.read_csv("rain_67max_gpp_april7.csv")

df_in = pd.read_csv("gs_67lai_climseas_april9.csv")#[["SITE_ID","EVI2","date"]]
rain_data = pd.read_csv("rain_67lai_climseas_april9.csv")

#df_in = pd.read_csv("gs_80lai_max_april10.csv")
#rain_data = pd.read_csv("rain_80lai_max_april10.csv")

df_in = pd.merge(df_in,df_base[["SITE_ID","Aridity","Aridity_gs","mat_data","map_data"]],on="SITE_ID",how="left")

#df_in = pd.read_csv("gs_50_mar28_complete.csv")
#df_in = pd.read_csv("gs_67_mar30_complete.csv")
#rain_data = pd.read_csv("rain_67_mar31.csv")

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
              "CRO":"Crop"
              }
biome_list = ["Evergreen needleleaf forest", "Mixed forest", "Deciduous broadleaf forest", "Evergreen broadleaf forest",
              "Grassland","Shrubland","Savanna","Crop"]

df_in["combined_biome"] = [simple_biomes[x] for x in df_in["IGBP"]]

#%%

#df_in = df_in.loc[df_in["gpp"]  > 0]
#df_in = df_in.loc[df_in["par"]  > 150]
#%%
#df_in["drel_spring"] = -np.clip(df_in["doy"] - df_in["summer_peak"],-np.inf,0) / (df_in["summer_peak"] - df_in["summer_start"])
#df_in["drel_fall"] = np.clip(df_in["doy"] - df_in["summer_peak"],0,np.inf) / (df_in["summer_end"] - df_in["summer_peak"])
#df_in["drel_both"] = -df_in["drel_spring"] + df_in["drel_fall"]
#%%
#df_in = df_in.loc[np.isfinite(df_in.drel_both)]
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
#%%

def get_lens(x,c):
    x2 = 1*x
    x2[0] = c+1
    day_diff = np.diff(np.where(x2 > c)[0])
    return day_diff[day_diff >= 1]
#%%
rain_sites = pd.unique(df_in.SITE_ID)
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
        if len(z) < 10:
            continue
        #halflen = int(len(z)/2)
        #z = z[halflen:]
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

def estGPP(df0):
    df = df0.copy()
    df["cond_norm"] = df.cond/df.LAI/df.dayfrac
    df["gpp_norm"] = df.gpp/df.LAI/df.dayfrac

    df["par_norm"] = df.par/df.dayfrac
    hicond = np.nanquantile(df["cond_norm"],0.75)
    dfhi = df.loc[df.cond_norm > hicond]
    himod = smf.ols("gpp_norm ~ par_norm + airt + C(year_new)",data=dfhi,missing="drop").fit()
    valid_years = pd.unique(dfhi.year_new)
    df = df.loc[df.year_new.isin(valid_years)].copy()
    df["gppmax"] = himod.predict(df)
    
    def tofit(k):
        interm = df.cond_norm/df.gppmax*k
        gpp_pred = df.LAI*df.dayfrac*df.gppmax*(1-np.exp(-interm))
        return np.nanmean((gpp_pred-df.gpp)**2)
    myopt = scipy.optimize.minimize_scalar(tofit)
    slope0 = myopt.x
    interm = df.cond_norm/df.gppmax*slope0
    df["kgpp"] = df.LAI*df.dayfrac*df.gppmax/slope0
    df["gpp_pred"] = df.LAI*df.dayfrac*df.gppmax*(1-np.exp(-interm))
    df["gppR2"] = r2_skipna(df["gpp_pred"],df["gpp"])
    return df, himod, slope0, valid_years
#%%site
def range_overlap(x,years):
    ystart = np.sort(pd.unique(years))[1:]
    ycor = []
    #yp = []
    for ymin in ystart:
        ypred = (years >= ymin).astype(float)
        ycor.append(cor_skipna(ypred,x)[0]**2)
        #yp.append(cor_skipna(ypred,x)[1])

    return ystart, ycor #, yp

#%%
bigyear = pd.read_csv("all_yearsites.csv")
# bigyear2 = pd.read_csv("AU_samplesites.csv")
# bigyear3 = pd.read_csv("US_samplesites.csv")

# bigyear = pd.concat([bigyear1,bigyear2,bigyear3]).reset_index()

bigyear = pd.merge(bigyear,bif_forest,on="SITE_ID",how='left')

bigyear["combined_biome"] = [simple_biomes[x] for x in bigyear["IGBP"]]

#%%
# goodsites = ['AU-Ade', 'AU-Cpr', 'AU-Dry', 'AU-Emr', 'AU-Gin', 'AU-Stp',
#        'AU-TTE', 'CA-Oas', 'CA-TP4', 'CN-Du2', 'ES-LJu', 'IT-Noe',
#        'NL-Loo', 'SD-Dem', 'US-AR1', 'US-AR2', 'US-Blo', 'US-Cop',
#        'US-Me2', 'US-Me3', 'US-Me5', 'US-Me6', 'US-NR1', 'US-SRC',
#        'US-SRG', 'US-SRM', 'US-UMB', 'US-UMd', 'US-Var', 'US-Whs',
#        'US-Wkg']
goodsites = ['AU-ASM', 'AU-DaS', 'AU-Dry', 'AU-Emr', 'AU-GWW', 'AU-Gin',
       'CN-Cng', 'ES-LJu', 'FI-Hyy', 'IT-Noe', 'IT-SRo', 'NL-Loo',
       'SD-Dem', 'US-AR1', 'US-AR2', 'US-Blo', 'US-Cop', 'US-Goo',
       'US-IB2', 'US-Me2', 'US-Me3', 'US-Me5', 'US-Me6', 'US-NR1',
       'US-SRC', 'US-SRG', 'US-SRM', 'US-Ton', 'US-Var', 'US-Whs',
       'US-Wkg']

#%%
all_results = []

#for site_id in pd.unique(df_in.SITE_ID)[:]:#[forest_daily[x] for x in [70,76]]:
#for site_id in goodsites:#[forest_daily[x] for x in [70,76]]:
for site_id in pd.unique(bigyear.SITE_ID):
    #%%
    if site_id=="ZM-Mon":
        continue
#%%
    print(site_id)
    dfgpp = df_in.loc[df_in.SITE_ID==site_id].copy()
    dfull = bigyear.loc[bigyear.SITE_ID==site_id].copy()
    #%%
    if np.max(dfull.LAI) < 0.05:
        continue
    #%%
    dfull["LAI"] = np.clip(dfull["LAI"],0.05,np.inf)

    dclim = dfull.groupby("doy").mean(numeric_only=True).reset_index()
    gpp_clim = np.array(dclim.gpp)
    #%%
    # gpp_adjoin = np.tile(gpp_clim,3)
    
    # gpp_clim_smooth = np.zeros(len(gpp_adjoin))
    # swidth = 14
    
    # for i in range(swidth,len(gpp_adjoin)-swidth):
    #     gpp_clim_smooth[i] = np.nanmean(gpp_adjoin[i-swidth:i+swidth+1])

    # gpp_clim_smooth[:swidth] = np.mean(gpp_clim[:swidth])
    # gpp_clim_smooth[-swidth:] = np.mean(gpp_clim[-swidth:])  
    
    # gpp_clim_smooth = gpp_clim_smooth[366:(366*2)]
    # #%%
    # gpp_clim_std =  gpp_clim_smooth / np.nanmax(gpp_clim_smooth)
    
#%%
    gpp_clim_std = np.array(dclim.LAI)/np.nanmax(dclim.LAI)
    
    #gpp_clim_std = (np.array(dclim.LAI)-np.nanmin(dclim.LAI)) / (np.nanmax(dclim.LAI)-np.nanmin(dclim.LAI))
    
    topday = np.argmax(gpp_clim_std)
    under50 = np.where(gpp_clim_std < 0.8)[0]
    try:
        summer_start = under50[under50 < topday][-1]
    except:
        summer_start = 0
    try:
        summer_end = under50[under50 > topday][0]
    except:
        summer_end = 365
    #%%
    dfull["is_summer"] = (dfull.doy >= summer_start)*(dfull.doy <= summer_end)
#%%
    #dfull["is_summer"] = dfull.date.isin(dfgpp.date)
    dfull = dfull.loc[dfull.rain_qc == 1].copy()
    #dfgpp = dfull.copy()
    #
    #dfgpp = dfgpp.loc[dfgpp.EVIside >= 0]
#    dfgpp = dfgpp.loc[dfgpp.drel_both >= 0]
#    dfgpp = dfgpp.loc[np.abs(dfgpp.drel_both) <= 0.5]
#%%
    #dfgpp["waterbal"] = 1*dfgpp.smc_lag
    #dfgpp = dfgpp.loc[np.isfinite(dfgpp.LAI)].copy()
    #dfgpp = dfgpp.loc[dfgpp.LAI > 0].copy()
    #%%
    if len(dfull) <= 25:
        continue
    if len(dfgpp) <= 25:
        continue
    #if dfgpp.inflow.iloc[0] > 0:
    #   continue
    #%%
    y1 = np.array(1-dfgpp.NIGHT)
    x1 = np.array(dfgpp.doy*2*np.pi/365)
    xstack = sm.add_constant(np.stack((np.cos(x1),np.sin(x1)),1))
    r_day = sm.OLS(y1,xstack).fit()
    dfgpp["dayfrac"] = r_day.fittedvalues
    dfgpp.dayfrac /= np.max(dfgpp.dayfrac)
    laimax = np.nanmax(dfgpp.LAI)
    dfgpp.LAI /= np.max(dfgpp.LAI)
    #%%
    
    x1 = np.array(dfull.doy*2*np.pi/365)
    xstack = sm.add_constant(np.stack((np.cos(x1),np.sin(x1)),1))
    dfull["dayfrac"] = r_day.predict(xstack)
    dfull["dayfrac"] /= np.nanmax(dfull["dayfrac"])
    
    dfull["LAI"] /= np.nanmax(dfull["LAI"])
    #%%
    dfgpp.cond = dfgpp.ET/dfgpp.vpd*100
    dfull.cond = dfull.ET/dfull.vpd*100
    
    #dfgpp0 = dfgpp.copy()
    #%%
#     dfgpp = dfgpp.loc[dfgpp.airt > 10]
#     dfgpp = dfgpp.loc[dfgpp.par > 100]
#     #%%
#     #dfgpp.LAI = (dfgpp.EVIint - np.mean(dfgpp.EVIint))/np.std(dfgpp.EVIint)*np.std(dfgpp.LAI) + np.mean(dfgpp.LAI)
# #%%
#     cn = 1*dfgpp.cond
#     cn -= np.mean(cn)
#     dfgpp = dfgpp.loc[np.abs(cn) < 3*np.std(cn)]
    
#     lcn = np.log(dfgpp.cond)
#     lcn -= np.mean(lcn)
#     dfgpp = dfgpp.loc[np.abs(lcn) < 3*np.std(lcn)]
   
    #%%
    # dfgpp = dfgpp.loc[np.isfinite(dfgpp.gpp)].copy()
    # dfgpp = dfgpp.loc[dfgpp.rain==0].copy()
    # dfgpp = dfgpp.loc[dfgpp.vpd > 0.5].copy()

    #%%
    #dfgpp.LAI = 1*dfgpp.EVI2
    #dfgpp.LAI /= np.max(dfgpp.LAI)
    #dfgpp["normcond"] = dfgpp.cond/dfgpp.LAI/dfgpp.dayfrac
    #dfgpp["normgpp"] = dfgpp.gpp/dfgpp.LAI/dfgpp.dayfrac
    #dfgpp["normrad"] = dfgpp.par/dfgpp.dayfrac
#    goodind = np.isfinite(dfull.cond*dfull.gpp)*(dfull.LAI > 0.25*np.nanmax(dfull.LAI))*(dfull.rain==0)*(dfull.rain_prev==0)*(dfull.vpd > 0.5)
    goodind = np.isfinite(dfull.cond*dfull.gpp)*(dfull.is_summer)*(dfull.rain==0)*(dfull.rain_prev==0)#*(dfull.vpd > 0.5)
#%%
    dfull2 = dfull.loc[goodind].copy()
    
    dlin = fit_gpp_linear2(dfull2.copy())
    dexp, maxmod, k0, goodyears = estGPP(dfull2.copy())
    
    dfull["summer_start"] = summer_start
    dfull["summer_end"] = summer_end
    #%%
    dfull["gppR2_exp"] = dexp.gppR2.iloc[0]
    dfull["gppR2_lin"] = dlin.gppR2.iloc[0]
    dfull["par_norm"] = dfull.par/dfull.dayfrac
    dfull = dfull.loc[dfull.year_new.isin(goodyears)].copy()
    dfull["gppmax_norm"] = maxmod.predict(dfull)
    dfull["kgpp"] = dfull.LAI*dfull.dayfrac*dfull["gppmax_norm"]/k0
    #dfull["cor_gpp_pval"] = cor_skipna(dfull2.cond/dfull2.kgpp,dfull2.gpp/dfull2.gppmax).pvalue
    #%%
    #smin_mm = -500
    #tauDay = 50
    dfGS = dfull.loc[dfull.is_summer].copy()

#    dfGS = dfull.copy()

#%%
    
    ddreg_fixed = []
    #ddreg_random = []
    et_over_dd = []
    ymaxes = []
    ymeans= []
    
    ymaxes0 = []
    ymeans0 = []
    # ddreg_fixed2 = []
    # et_over_dd2 = []
    
    ddlabel = []
    ddii = 0
    
    grec = []
    frec = []
    et_plain = []
    vpd_plain = []
    etcum = []
    # doy_label = []
    # year_label=[]
    
    # tlist = []
    # tse = []
    
    # xylist = []
    # yylist = []
    
    # dparlist = []
    # dselist = []
    # dpval = []
    
    # etrec = []
    # frec = []
    #%%
    for y0 in pd.unique(dfGS.year_new):
    #%%
        dfy = dfGS.loc[dfGS.year_new==y0].copy()
        if np.sum(np.isfinite(dfy.ET)) < 10:
            continue
        #mederr = np.nanmedian(dfy.gpp / dfy.gpp_pred)
        #dfy.kgpp = 1.8/45*dfy.LAI
        #dfy["gmax"] = 4*dfy.LAI
        #dfy.kgpp *= 1.3
        #doy_arr = np.arange(dfy.doy.iloc[0],dfy.doy.iloc[-1]+1)
        doy_indata = np.array(dfy.doy)
        vpd_arr = np.array(dfy.vpd)/100
        vpd_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(vpd_arr)],
                            vpd_arr[np.isfinite(vpd_arr)])
        k_mm_day = np.array(dfy.kgpp)*mol_s_to_mm_day #* np.array(dfy.gpp/dfy.gpp_pred)
        rain_arr = np.array(dfy.rain)
        
        #%%
    #et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        
        et_mmday = np.array(dfy.ET)*mol_s_to_mm_day
        
        #et_mmday = np.array(dfy.cond)*mol_s_to_mm_day
        
        et_mmday_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(et_mmday)],
                            et_mmday[np.isfinite(et_mmday)])
        # w_arr = np.array(dfy.waterbal)
        
        
        #et_mmday[dfy.airt < 10] = np.nan
        #et_mmday[dfy.par < 100] = np.nan

        #et_mmday[dfy.vpd < 0.5] = np.nan
        et_mmday[dfy.ET_qc < 0.5] = np.nan
        
#%%
        rain_d5 = np.array([0] + list(np.where(rain_arr > 5)[0]) + [len(rain_arr)])
        dd5= np.diff(rain_d5)
        ymaxes.append(np.max(dd5))
        ymeans.append(np.mean(dd5[dd5 >= 2]))

#%%
        rain_days = np.array([0] + list(np.where(rain_arr > 0)[0]) + [len(rain_arr)])
        ddgood = np.where(np.diff(rain_days) >= 7)[0]
        
        ddstart = rain_days[ddgood]+2
        ddend = rain_days[ddgood+1]
    
        etnorm = et_mmday**2 / (vpd_arr*k_mm_day)
        
        #etnorm[dfy.airt < 10] = np.nan
        #etnorm[dfy.par < 100] = np.nan

        #etnorm[dfy.vpd < 0.5] = np.nan
        #etnorm[dfy.ET_qc < 0.5] = np.nan
        
        doyY = np.array(dfy.doy)
    #%%
        dd0= np.diff(rain_days)
        ymaxes0.append(np.max(dd0))
        ymeans0.append(np.mean(dd0[dd0 >= 2]))
    #%%
        #tau_with_unc = []
        #winit_with_unc = []
        for ddi in range(len(ddstart)):
            
            f_of_t = (vpd_arr*k_mm_day)[ddstart[ddi]:ddend[ddi]]
#           # g_of_t = np.cumsum(np.sqrt(f_of_t))
#            g_of_t = np.array([0] + list(np.cumsum(np.sqrt(f_of_t))))[:-1]
            g_of_t = np.array([0] + list(np.cumsum(np.sqrt(f_of_t))))[:-1]
            #g_of_t = g_of_t[:20]
            yfull = et_mmday[ddstart[ddi]:ddend[ddi]]/np.sqrt(f_of_t)
            #yfull = yfull[:20]
            
            doyDD = doyY[ddstart[ddi]:ddend[ddi]]
            
            #if r1.params[1] < 0 and r1.pvalues[1] < 0.05:
            if np.sum(np.isfinite(yfull*g_of_t)) >= 5: # and doyDD[0] > 150:
                #rdd = sm.OLS(yfull,sm.add_constant(g_of_t),missing='drop').fit()
                #if rdd.params[1] < 0 and rdd.pvalues[1] < 0.05:
                et_over_dd.append(yfull - np.nanmean(yfull))
                ###et_over_dd.append((yfull - np.nanmean(yfull))/np.std(g_of_t))
                ddreg_fixed.append(g_of_t - np.mean(g_of_t[np.isfinite(yfull)]))
                ddlabel.append([ddii]*len(yfull))
                
                frec.append(f_of_t)
                grec.append(g_of_t)
                vpd_plain.append(vpd_arr[ddstart[ddi]:ddend[ddi]])
                et_plain.append(et_mmday[ddstart[ddi]:ddend[ddi]])
                
                # yfull = etnorm[ddstart[ddi]:ddend[ddi]]#[:20]
                etsel = et_mmday_interp[ddstart[ddi]:ddend[ddi]]#[:20]
                etcum.append(np.array([0] + list(np.cumsum(etsel)))[:-1])
                # et_over_dd.append(yfull - np.nanmean(yfull))
                # ddreg_fixed.append(etcum - np.mean(etcum[np.isfinite(yfull)]))
                ddii += 1
        #%%
    if len(ddreg_fixed) == 0:
        continue
#     #%%
    row0 = np.concatenate(ddreg_fixed)
    et_topred = np.concatenate(et_over_dd)
    if np.sum(np.isfinite(et_topred*row0)) < 10:
        continue
    #%%
    r1= sm.OLS(et_topred,row0,missing='drop').fit()
    if r1.pvalues[0] > 0.05 or r1.params[0] > 0:
        continue
    
    #%%
    dfull["tau_ddreg"] = -1/r1.params[0]
    dfull["tau_ddreg_lo"] = -1/(r1.params[0] - 2*r1.bse[0])
    dfull["tau_ddreg_hi"] = -1/(r1.params[0] + 2*r1.bse[0])
    #%%
    dfull["tau_rel_err"] = -r1.bse[0]/r1.params[0]
    #dfull = dfull.loc[dfull.year_new.isin(pd.unique(dfgpp0.year_new))].copy()
    #dfull["dayfrac"] = (1-dfull.NIGHT)
    dfull["seas_rain_mean5"] = np.mean(ymeans)
    dfull["seas_rain_max5"] = np.mean(ymaxes)
    dfull["seas_rain_mean0"] = np.mean(ymeans0)
    dfull["seas_rain_max0"] = np.mean(ymaxes0)
    #%%
    btab = pd.DataFrame({"ddi":np.concatenate(ddlabel),
                         "G":np.concatenate(grec),
                         "ET":np.concatenate(et_plain),
                         "et_per_F_dm":et_topred,
                         "F":np.sqrt(np.concatenate(frec)),
                         "VPD":np.concatenate(vpd_plain),
                         "etcum":np.concatenate(etcum)})
    #%%
    btab["cond"] = btab.ET/btab.VPD
    tau = -1/r1.params[0]
    #%%
    btab["term1"] = -1/tau*btab.G*btab.F
    #%%
    ddmean = btab.groupby("ddi").mean().reset_index()
    ddmean["b"] = (ddmean.ET - ddmean.term1)/ddmean.F
    #%%
    btab = pd.merge(btab,ddmean[["ddi","b"]],on='ddi',how='left')
#%%
    btab["ETpred"] = btab.term1 + btab.b*btab.F
    dfull["etr2_b"] = r2_skipna(btab.ETpred,btab.ET)
    dfull["gr2_b"] = r2_skipna(btab.ETpred/btab.VPD,btab.cond)
    #%%
    
    ddfirst = btab.groupby("ddi").first().reset_index()
    #ddfirst["aETinit"] = 1*ddfirst["ET"]/ddfirst["F"]
    ddfirst["ETinit"] = 1*ddfirst["ET"]
    ddfirst["Finit"] = 1*ddfirst["F"]
    btab = pd.merge(btab,ddfirst[["ddi","ETinit","Finit"]],on='ddi',how='left')
    #%%
    btab["b_init"] = (btab.ETinit/btab.Finit)
#%%
    # dfgpp0["etpred_newmod_mmday"] = etrec
    # dfi = pd.merge(dfi,dfgpp0[["date","etpred_newmod_mmday"]],on="date",how="left")
    # dfi["etr2_reg"] = 1- np.nanmean((dfgpp0.ET-dfgpp0.etpred_newmod_mmday/mol_s_to_mm_day)**2)/np.nanvar(dfgpp0.ET)
    # dfi["gr2_reg"] = 1- np.nanmean((dfgpp0.cond-dfgpp0.etpred_newmod_mmday/mol_s_to_mm_day/(dfgpp0.vpd/100))**2)/np.nanvar(dfgpp0.cond)
#%%
    all_results.append(dfull)
#%%
    
#%%
all_results = pd.concat(all_results)
#%%

#%%
site_count = np.array(all_results.groupby("SITE_ID").count()["waterbal"])
site_year = np.array(all_results.groupby("SITE_ID").nunique()["year"])

#%%
df1 = all_results.groupby("SITE_ID").first().reset_index()

#%%
def qt_gt1(x,q):
    return np.quantile(x[x >= 1],q)
def mean_gt1(x):
    return np.mean(x[x >= 1])
#%%


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
df1 = pd.merge(df1,rain_site_tab,on="SITE_ID",how="left")


#%%
df_meta = df1.copy()

#fval = ((1-df_meta.etr2_null)-(1-df_meta.etr2_smc))/(1-df_meta.etr2_smc)*(df_meta.npoints-4)
#df_meta["ftest"] = 1-scipy.stats.f.cdf(x=fval,dfn=1,dfd=df_meta.npoints-4)
#df_meta = df_meta.loc[df_meta.ftest < 0.01]
#df_meta = df_meta.loc[df_meta.LOCATION_LAT > 0]
#df_meta = df_meta.loc[df_meta.tau_rel_unc < 0.25].copy()
udf = all_results.groupby("SITE_ID").nunique().reset_index()
udf["nyears"] = 1*udf["year"]
df_meta = pd.merge(df_meta,udf[["SITE_ID","nyears"]],on="SITE_ID",how="left")
#df_meta = df_meta.loc[df_meta.gppR2 > 0.01].copy()

#df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_no_cond > 0.01]
#df_meta = df_meta.loc[df_meta.gppR2-df_meta.gppR2_only_cond > 0.01]

#df_meta = df_meta.loc[df_meta.etr2_smc > 0]

#df_meta = df_meta.loc[df_meta.tau > 0]
#df_meta = df_meta.loc[df_meta.tau_lo > 0]
#df_meta = df_meta.loc[df_meta.tau_hi > 0]


#df_meta["rel_err"] = (df_meta.etr2_smc-df_meta.etr2_null)#/(1-df_meta.etr2_null)
#df_meta = df_meta.loc[df_meta.rel_err > 0.05]
#df_meta = df_meta.loc[df_meta.ancond_lo_smc/df_meta.ancond_hi_smc < 0.75]
#df_meta = df_meta.loc[df_meta.bcond_lo_smc/df_meta.bcond_hi_smc < 0.67]

#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Fire"]
#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Agriculture"]
#df_meta = df_meta.loc[df_meta.DOM_DIST_MGMT != "Grazing"]

#df_meta = df_meta.loc[df_meta.nE_lo_smc/df_meta.nE_hi_smc < 0.8]
#df_meta = df_meta.loc[df_meta.max_limitation < 0.9]
#df_meta["soil_rel_max"] = np.clip(df_meta.soil_max/1000 - df_meta.smin,0,100)
#df_meta["soil_rel_min"] = np.clip(df_meta.soil_min/1000 - df_meta.smin,0,100)
#df_meta["soil_ratio"] = df_meta["soil_rel_min"]/df_meta["soil_rel_max"]
#%%
df_meta = pd.merge(df_meta,metadata,left_on="SITE_ID",right_on="fluxnetid",how="left")
#%%
#df_meta = df_meta.loc[df_meta["cor_gpp_pval"] < 0.05]
#%%
df_meta = df_meta.loc[df_meta.gppR2_exp - df_meta.gppR2_lin > 0]

#df_meta = df_meta.loc[df_meta.gppR2_base - df_meta.gppR2_only_cond > 0.05]
#df_meta = df_meta.loc[df_meta.gppR2_base - df_meta.gppR2_no_cond > 0.05]
#df_meta = df_meta.loc[df_meta.inflow == 0]
df_meta = df_meta.loc[df_meta.tau_ddreg_lo > 0]
df_meta = df_meta.loc[df_meta.tau_ddreg_hi > 0]
#%%
#df_meta = df_meta.loc[(df_meta.tau_hi-df_meta.tau_lo)/df_meta.tau_reg < 0.75]
df_meta = df_meta.loc[df_meta.tau_rel_err < 0.25]

#df_meta = df_meta.loc[(df_meta.tau_ddreg_hi-df_meta.tau_ddreg_lo)/ df_meta.tau_ddreg < 1]
#%%
df_meta["ddrain_mean"] = 1*df_meta.seas_rain_max5
df_meta["ddrain_2mean"] = 1*df_meta.seas_rain_mean5
#%%
df_meta["gsrain_len"] = df_meta.summer_end - df_meta.summer_start
#%%
rainmod = smf.ols("tau_ddreg ~ ddrain_mean",data=df_meta).fit()
#%%
r2_11 = 1-np.mean((df_meta.ddrain_mean-df_meta.tau_ddreg)**2)/np.var(df_meta.tau_ddreg)
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
        pointI, = ax.plot(subI.ddrain_mean,subI.tau_ddreg,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
xmax = np.max(df_meta.ddrain_mean)
ymax = np.max(df_meta.tau_ddreg)


ax.set_xlim(0,1.1*xmax)
ax.set_ylim(0,1.1*ymax)
ax.set_xlabel("Annual-mean $D_{max}$ (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")
#%%
#allrain = pd.read_csv("year_round_rain_stats.csv")
#allrain = pd.read_csv("gs_start_rain_stats_maxLAI.csv")
#df_meta = pd.merge(df_meta,allrain,on="SITE_ID",how='left')

#%%
# yscaler = np.sqrt(zsoil_mol)
# molm2_to_mm = 18/1000
# s2day = 60*60*24

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
df_meta = pd.merge(df_meta,df_base[["SITE_ID","Aridity","Aridity_gs","mat_data","map_data"]],on="SITE_ID",how="left")


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


#%%
df_meta3 = df_meta.sort_values("etr2_b")
df_meta3["et_rank"] = np.arange(len(df_meta3))

fig,axes = plt.subplots(3,1,figsize=(15,10))
ax = axes[0]

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.et_rank,subI.etr2_b,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
ax.set_xticks(df_meta3.et_rank,df_meta3.SITE_ID,rotation=90)
#ax.set_xlim(0,250)
ax.set_ylim(0,1)
#ax.set_xlabel("Rank",fontsize=24)
ax.set_ylabel(r"$R^2$ of ET",fontsize=24)

#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

df_meta3 = df_meta.sort_values("gr2_b")
df_meta3["g_rank"] = np.arange(len(df_meta3))
ax = axes[1]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.g_rank,subI.gr2_b,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.g_rank,df_meta3.SITE_ID,rotation=90)
ax.set_ylabel(r"$R^2$ of g",fontsize=24)
#ax.axhline(0,color='k')


df_meta3 = df_meta.sort_values("gppR2_b")
df_meta3["gpp_rank"] = np.arange(len(df_meta3))
ax = axes[2]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.gpp_rank,subI.gppR2_b,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.gpp_rank,df_meta3.SITE_ID,rotation=90)
ax.set_ylabel(r"$R^2$ of GPP",fontsize=24)
fig.tight_layout()
fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.02),ncols=2)
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%
# plt.figure(figsize= (10,10))
# plt.subplot(2,1,1)
# plt.plot(all_results.drel_both, np.log(all_results.gpp/all_results.gpp_pred),'.',alpha=0.1); 
# plt.ylim(-1,1);
# plt.xlim(-2,2);

# plt.ylabel("log GPP error")
# plt.subplot(2,1,2)
# plt.plot(all_results.drel_both, np.log(all_results.ET/all_results.et_tau),'.',alpha=0.1); 
# plt.ylim(-1,1);
# plt.xlim(-2,2);
# plt.ylabel("log ET error")

# plt.xlabel("Day relative to growing season peak")
#%%
# lerr_gpp = np.log(all_results.gpp/all_results.gpp_pred)
# lerr_et = np.log(all_results.ET/all_results.et_tau)
# ok_err = (np.abs(lerr_gpp) < 0.5)*(np.abs(lerr_et) < 0.5)
# print(cor_skipna(lerr_gpp[ok_err],lerr_et[ok_err]))
#%%
# plt.figure()
# plt.plot(lerr_gpp, lerr_et,'.',alpha=0.1); 
# plt.ylim(-1,1)
# plt.xlim(-1,1)
# plt.xlabel("log GPP error")
# plt.ylabel("log ET error")

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
df_meta["tau"] = 1*df_meta.tau_ddreg
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
all_results = pd.merge(all_results,rain_site_tab,on="SITE_ID",how="left")
#%%
# Blo_tab = pd.DataFrame({"ET":etrec[17],"v_gA":frec[17]})
# #%%
# mmstab = pd.DataFrame({"ET":etrec[5],"v_gA":frec[5]})
# # #%%
#%%
def nanterp(x):
    xind = np.arange(len(x))
    return np.interp(xind,
                      xind[np.isfinite(x)],
                      x[np.isfinite(x)])

#%%
# plt.figure(figsize = (12,8))
# plt.subplot(2,2,1)
# plt.plot(Blo_tab.ET,'o-',label="US-Blo"); 
# plt.plot(mmstab.ET,'o-',label="CA-TP4");
# plt.xlabel("Day of drydown")
# plt.ylabel("ET (mm/day)")
# plt.legend()

# plt.subplot(2,2,2)
# plt.plot(Blo_tab.v_gA,'o-'); plt.plot(mmstab.v_gA,'o-');
# plt.xlabel("Day of drydown")
# plt.ylabel("$g_A * VPD$ (mm/day)")


# plt.subplot(2,2,3)
# plt.plot(Blo_tab.ET**2/Blo_tab.v_gA,'o-'); 
# plt.plot(mmstab.ET**2/mmstab.v_gA,'o-');
# plt.ylabel("$ET^2 / (g_A * VPD)$ (mm/day)")
# plt.xlabel("Day of drydown")

# plt.subplot(2,2,4)

# etcum1 = np.array([0] + list(np.cumsum(nanterp(Blo_tab.ET))))[:-1]
# etcum2 = np.array([0] + list(np.cumsum(nanterp(mmstab.ET))))[:-1]
# reg1 = sm.OLS(Blo_tab.ET**2/Blo_tab.v_gA,sm.add_constant(etcum1),missing='drop').fit()
# reg2 = sm.OLS(mmstab.ET**2/mmstab.v_gA,sm.add_constant(etcum2),missing='drop').fit()
# xarr = np.array([0,np.max(etcum1)])
# plt.plot(etcum1,Blo_tab.ET**2/Blo_tab.v_gA,'o-');
# plt.plot(xarr,xarr*reg1.params[1] + reg1.params[0],'k--',label=r"$\tau$ = 52 days")
# plt.plot(etcum2,mmstab.ET**2/mmstab.v_gA,'o-');
# xarr = np.array([0,np.max(etcum2)])
# plt.plot(xarr,xarr*reg2.params[1] + reg2.params[0],'k:',label=r"$\tau$ = 14 days")
# plt.legend()

# plt.ylabel("$ET^2 / (g_A * VPD)$ (mm/day)")
# plt.xlabel("Cumulative ET (mm)")

# plt.tight_layout()
#%%
# j = -1
# fj = frec[j][30:]; ej = etrec[j][30:];
# #f2 = sqrt(f)
# #et ~ -1/tau * cumsum(f2)*f2 + f2
# f2 = np.sqrt(fj)
# g2 = np.array([0] + list(np.cumsum(f2)))[:-1]
# #%%

# #%%
# tau = 15
# term1 = -1/tau*g2*f2
# c2 = (np.nanmean(ej) - np.nanmean(term1))/np.nanmean(f2)
# epred20 = term1 + c2*f2

# tau = 60
# term1 = -1/tau*g2*f2
# c2 = (np.nanmean(ej) - np.nanmean(term1))/np.nanmean(f2)
# epred50 = term1 + c2*f2

# #%%
# plt.figure()
# plt.plot(ej,'ko-',linewidth=3,label="Eddy covariance")
# plt.plot(epred50,'o-',linewidth=3,alpha=0.6,label="Model, $tau$ = 60 days")
# plt.plot(epred20,'o-',linewidth=3,alpha=0.6,label="Model, $tau$ = 15 days")
# plt.legend()
# plt.xlabel("Day of drydown",fontsize=22)
# plt.ylabel("ET (mm/day)",fontsize=22)
# sinf = (ej / np.sqrt(fj) / np.sqrt(2/tau))**2
# s = sinf[0]
# sirec = np.zeros(len(fj))
# for i in range(len(fj)):
#     sirec[i] = s
#     eti = np.sqrt(2/tau*s*fj[i])
#     s -= eti
# etrec = np.sqrt(sirec * 2/tau * fj)
#%%
# tau = 50
# #sinf = (ej / np.sqrt(fj) / np.sqrt(2/tau))**2
# s = sinf[0]
# sirec = np.zeros(len(fj))
# for i in range(len(fj)):
#     sirec[i] = s
#     eti = np.sqrt(2/tau*s*fj[i])
#     s -= eti
# etrec2 = np.sqrt(sirec * 2/tau * fj)
#%%
# Blo_tab = pd.DataFrame({"x":row0,"y":et_topred})
# #%%
# TP4_tab = pd.DataFrame({"x":row0,"y":et_topred})
# #%%
# plt.figure()
# plt.plot(Blo_tab.x,Blo_tab.y,'.',label=r"US-Blo, $\tau$ = 94 days")
# plt.plot(TP4_tab.x,TP4_tab.y,'o',label=r"CA-TP4, $\tau$ = 8 days")
# plt.xlabel("De-meaned G, $(mm*day)^{0.5}$")
# plt.ylabel("De-meaned ET/F, $(mm/day)^{0.5}$")
# xarr = np.array([-100,100])
# r1 = sm.OLS(Blo_tab.y,Blo_tab.x,missing='drop').fit()
# plt.plot(xarr,r1.params[0]*xarr,'--',color="blue",linewidth=2)

# xarr = np.array([-10,10])
# r2 = sm.OLS(TP4_tab.y,TP4_tab.x,missing='drop').fit()
# plt.plot(xarr,r2.params[0]*xarr,'--',color="red",linewidth=2)
# plt.ylim(-1.5,1.5)
# plt.legend()
#%%
#%%
#btab["c"] = btab.b/np.sqrt(2/tau)
#%%
#btab["S_est"] = 0.25*(-)

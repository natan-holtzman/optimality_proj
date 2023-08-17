# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:54:18 2023

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
#%%
bif_data = pd.read_csv("fn2015_bif_tab.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])]
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])]
metadata = pd.read_csv(r"C:\Users\nholtzma\Downloads\fluxnet_site_info_all.csv")

all_daily = glob.glob(r"C:\Users\nholtzma\Downloads\fluxnet2015\daily_data\*.csv")
forest_daily = [x for x in all_daily if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
#%%

all_hh = glob.glob(r"C:\Users\nholtzma\Downloads\fluxnet2015\*_HH_*.csv")
forest_hh = [x for x in all_hh if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
all_hourly = glob.glob(r"C:\Users\nholtzma\Downloads\fluxnet2015\*_HR_*.csv")
forest_h = [x for x in all_hourly if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
forest_all = forest_hh + forest_h
#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%s
#latsort = bif_forest.sort_values("LOCATION_LAT")
#%%
def find_peaks(x):
    peaklocs = np.where((x[1:-1] >= x[:-2])*(x[1:-1] >= x[2:]))[0] + 1
    return peaklocs
#if there is only one peak, we're good
#if there are two peaks, check how much lower the middle trough is

def find_valleys(x):
    valleys = np.where((x[1:-1] <= x[:-2])*(x[1:-1] <= x[2:]))[0] + 1
    return valleys

def wrap_min(x):
    minday = np.argmin(x)
    x2 = 0*x
    chunk1 = x[:minday]
    chunk2 = x[minday:]
    x2[:len(chunk2)] = 1*chunk2
    x2[len(chunk2):] = 1*chunk1
    return x2


def make_1peak(x1):
    x2 = 0*x1
    topday = np.argmax(x1)
    x2[topday] = x1[topday]
    
    for i in range(topday+1,len(x1)):
        x2[i] = min(x2[i-1],x1[i])
    for i in range(topday-1,-1,-1):
        x2[i] = min(x2[i+1],x1[i])
    return x2
#test_sites = np.array(latsort.SITE_ID)[::5]
#%%
all_daily = pd.read_csv("all_yearsites_2gpp.csv",parse_dates=["date"])
#%%
site_tab = []
#%%
for site in bif_forest.SITE_ID:
    print(site)
    try:
        fname = [x for x in forest_all if site in x][0]
    except:
        continue
    #%%
    df = pd.read_csv(fname)
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"],format="%Y%m%d%H%M")
    df["date"] = df["TIMESTAMP_START"].dt.date
    df["hour"] = df.TIMESTAMP_START.dt.hour
    df["doy_raw"] = df.TIMESTAMP_START.dt.dayofyear
    
    df["year_raw"] = df.TIMESTAMP_START.dt.year
    df = df.loc[df.year_raw >= 2001].copy()
    #%%
    if len(df) < 25*24:
        continue
    #%%
    dfull = all_daily.loc[all_daily.SITE_ID==site].copy().drop(columns="summer_start")
    #%%
    dcount = dfull.groupby("year_new").count().reset_index()
    fullyear = dcount.year_new.loc[dcount.date > 300]
    dfull = dfull.loc[dfull.year_new.isin(fullyear)]
    #%%
    if np.max(dfull.LAI) < 0.05:
        continue
    if len(dfull) < 300:
        continue
    #%%
    if np.max(dfull.LAI) < 0.05:
        continue
    #%%
    dfull["LAI"] = np.clip(dfull["LAI"],0.05,np.inf)
    dfull["LAIdiff"] = [0] + list(np.diff(dfull.LAI))

    #gpp_clim = np.array(dclim.gpp)
    year95 = dfull.groupby("year_new").max(numeric_only=True).reset_index()

    #year95["gpp_y95"] = 1*year95["gpp_smooth"]
    year95["lai_y95"] = 1*year95["LAI"]
    dfull = pd.merge(dfull,year95[["year_new","lai_y95"]],how="left",on="year_new")
    dfull["LAI_gt50"] = dfull.LAI/dfull.lai_y95 > 0.5
    dfull["gpp"] = np.clip((dfull.gpp_nt + dfull.gpp_dt)/2,0,np.inf)
    #%%
    df[df == -9999] = np.nan
    
    df["PPFD_in"] = df.SW_IN_F
    df["VPD"] = df.VPD_F/10
    df["LE"] = np.clip(df["LE_F_MDS"],0,np.inf)
    g1 = np.clip(0.5*(df.GPP_NT_VUT_REF + df.GPP_DT_VUT_REF),0,np.inf)
    g1[df.GPP_NT_VUT_REF < 0] = np.nan
    g1[df.GPP_DT_VUT_REF < 0] = np.nan
    g1[df.LE == 0] = 0
    g1[df.PPFD_in == 0] = 0

    df["gpp"] = g1
    
    df["T_AIR"] = df.TA_F
    df["cond"] = df.LE/44200/(df.VPD/100)
#%%
    #df["LAI_gt50"] = (df.gpp_smooth/df.gpp_y95) > 0.67
    year_list = pd.unique(dfull.year_new)
    
    # gcor = []
    # for year in year_list:
    #     dfy = dfull.loc[dfull.year_new==year].reset_index()
    #     x = wrap_min(np.array(dfy.LAI)); x2 = make_1peak(x)
    #     gcor.append(np.corrcoef(x,x2)[0,1]**2)
    # goodyears = year_list[np.array(gcor) > 0.9]
    # dfull = dfull.loc[dfull.year.isin(goodyears)].copy()
    #%%
    # year_list = pd.unique(dfull.year_new)
    
    # npeaks = []
    # for year in year_list:
    #     dfy = dfull.loc[dfull.year_new==year].reset_index()
    #     x = np.array(dfy.LAI)
    #     x2 = (x-np.min(x))/(np.max(x)-np.min(x))
    #     npeaks.append(len(scipy.signal.find_peaks(x2,prominence=0.25)[0]))
    # goodyears = year_list[np.array(npeaks) <= 1]
    # #%%
    # dfull = dfull.loc[dfull.year_new.isin(goodyears)].copy()
    #%%
    gs_starts = []
    gs_ends = []
    for year in year_list:
        dfy = dfull.loc[dfull.year_new==year].reset_index()
        topday = np.argmax(dfy.LAI)
        
        yearclim = (dfy.LAI-np.nanmin(dfy.LAI))/(np.nanmax(dfy.LAI)-np.nanmin(dfy.LAI))
        #yearclim = (dfy.LAI)/(np.nanmax(dfy.LAI))
        #yearclim = (dfy.gpp)/(np.nanmax(dfy.gpp))

#        topday = np.argmax(dfy.gpp_smooth)
        under50 = np.where(yearclim < 0.75)[0]
        #under50 = np.where(~dfy.LAI_gt50)[0]
        try:
            summer_start = under50[under50 < topday][-1] + 1
        except:
            summer_start = 0
        try:
            summer_end = under50[under50 > topday][0] -1
        except:
            summer_end = 365
        gs_starts.append(summer_start)
        gs_ends.append(summer_end)
        
    summer_df = pd.DataFrame({"year_new":year_list,
                              "summer_start":gs_starts,
                              "summer_end":gs_ends})
    dfull= pd.merge(dfull,summer_df,on="year_new",how="left")
    
    is_summer = np.array((dfull.doy >= dfull.summer_start)*(dfull.doy <= dfull.summer_end))
    
#    gpp_clim_std = np.array(dclim.LAI-np.nanmin(dclim.LAI))/(np.nanmax(dclim.LAI)-np.nanmin(dclim.LAI))
    #%%
    if len(dfull) == 0:
        continue
    #%%
    dclim = dfull.groupby("doy").mean(numeric_only=True).reset_index()
#%%
    gpp_clim_std = np.array(dclim.LAI)/(np.nanmax(dclim.LAI))
    #gpp_clim_std = np.array(dclim.LAI-np.nanmin(dclim.LAI))/(np.nanmax(dclim.LAI)-np.nanmin(dclim.LAI))

    topday = np.argmax(gpp_clim_std)
    under50 = np.where(gpp_clim_std < 0.75)[0]
    try:
        clim_summer_start = under50[under50 < topday][-1] + 1
    except:
        clim_summer_start = 0
    try:
        clim_summer_end = under50[under50 > topday][0] -1
    except:
        clim_summer_end = 365
    dfull["clim_summer"] = (dfull.doy >= clim_summer_start)*(dfull.doy <= clim_summer_end)
    #is_summer *= clim_summer
    #%%
    # laidiff = np.array(np.diff(dclim.LAI))
    # laidiff[0] = 0
    # laidiff[-1] = 0
    # under50 = np.where(np.abs(laidiff) > np.quantile(np.abs(laidiff),0.75))[0]
    # #under50 = np.where(np.abs(laidiff) > 0.75*np.max(np.abs(laidiff)))[0]

    # try:
    #     clim_summer_start = under50[under50 < topday][-1] + 1
    # except:
    #     clim_summer_start = 0
    # try:
    #     clim_summer_end = under50[under50 > topday][0] -1
    # except:
    #     clim_summer_end = 365
    # dfull["clim_summer"] = (dfull.doy >= clim_summer_start)*(dfull.doy <= clim_summer_end)
    
    
    #%%
#    daily_gs = dfull.loc[is_summer*dfull.clim_summer].copy()
    
    daily_gs = dfull.loc[dfull.clim_summer].copy()
    
    #daily_gs = daily_gs.loc[daily_gs.LAI >= np.nanmedian(dfull.LAI)].copy()
    #daily_gs = dfull.loc[is_summer].copy()

    #daily_gs = daily_gs.loc[daily_gs.year_new >= 2001]
    
    daily_gs["date"] = daily_gs["date"].dt.date
    #%%
    # ydf = daily_gs.groupby("year").mean(numeric_only=True).reset_index()
    # yearlist = pd.unique(daily_gs.year)
    # tp = []
    # for y in yearlist:
    #     x_in = ydf.loc[ydf.year == y,"LAI"]
    #     x_out = ydf.loc[ydf.year != y,"LAI"]
    #     tp.append(((x_in-np.mean(x_out))/np.std(x_out)).iloc[0])
    # tp = np.array(tp)
    # goodyears = yearlist[np.abs(tp) < 3]
    #     #tp.append(scipy.stats.ttest_ind(x_in,x_out,equal_var=False).pvalue)
    # daily_gs = daily_gs.loc[daily_gs.year.isin(goodyears)]
    
    #%%
    df = pd.merge(df,daily_gs[["date","LAI","lai_y95"]],on="date",how="inner")
    #%%
    # dg = dfull.copy()
    # dg.loc[dfull.potpar < 0.5*np.max(dfull.potpar),"rain"] = 10
    # raingaps = {}
    # raingaps["ddlen"] = list(np.diff(np.where(np.array(dg.rain > 0))[0])) + [0]
    # raingaps["year"] = np.array(dg.year)[np.where(np.array(dg.rain > 0))[0]]
    # raingaps = pd.DataFrame(raingaps)
    # rainymax = raingaps.groupby("year").max(numeric_only=True).ddlen
    #%%
    dfday = df.loc[df.PPFD_in > 100].copy()
    dfday = dfday.loc[dfday.VPD > 0.5].copy()
    dfday = dfday.loc[dfday.LE > 0].copy()
    dfday = dfday.loc[dfday.P_F == 0].copy()

    dfday = dfday.loc[np.isfinite(dfday.gpp)].copy()
    dfday = dfday.loc[dfday.LE_F_MDS_QC <= 1]
    
    dfday = dfday.loc[np.isfinite(dfday.gpp)]
    
    dfday = dfday.loc[dfday.gpp > 0].copy()
    
    dfday = dfday.loc[dfday.NEE_VUT_REF_QC <= 1].copy()
    #%%
    relwue = dfday.gpp/dfday.cond
    dfday = dfday.loc[relwue < 500].copy()
#%%
    dfday = dfday.loc[dfday.year_raw >= 2001].copy()
#%%
    if len(dfday) < 25:
        continue
#%% 
    
    #%%
    dfday["cond_norm"] = dfday.cond/dfday.LAI
    dfday["gpp_norm"] = dfday.gpp/dfday.LAI

    dfhi = dfday.loc[dfday.cond_norm > np.quantile(dfday.cond_norm,0.9)].copy()
    #%%
    
    # def tofit(pars):
    #     amax1,kA,gmax1,kG = pars
        
    #     amax = amax1*dfday.PPFD_in/(dfday.PPFD_in + kA)
    #     gA = gmax1*dfday.PPFD_in/(dfday.PPFD_in + kG)

    #     gpp_pred = amax*(1-np.exp(-dfday.cond_norm/gA))
    #     z = (gpp_pred-dfday.gpp_norm)#[dfday.VPD > 1]
    #     return z
    # himean = np.mean(dfhi.gpp_norm)
    # fit0 = np.array([himean,300,himean/150,400])
    # myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))

    # amax1,kA,gmax1,kG = myfit.x
    # #%%
    # df["gA_hourly"] = gmax1*df.PPFD_in/(df.PPFD_in + kG) * df.LAI
    # df["amax_hourly"] = amax1*df.PPFD_in/(df.PPFD_in + kA) * df.LAI
    # df["gpp_pred_hourly"] = df["amax_hourly"] * (1 - np.exp(-df.cond/df["gA_hourly"]))
    #%%
    # bsres = []
    # for bsi in range(20):
    #     dfday2 = dfday.sample(len(dfday),replace=True)
    #     def tofit(pars):
    #         amax1,kA,gmax1,kG = pars
            
    #         amax = amax1*dfday2.PPFD_in/(dfday2.PPFD_in + kA)
    #         gA = gmax1*dfday2.PPFD_in/(dfday2.PPFD_in + kG)

    #         gpp_pred = amax*(1-np.exp(-dfday2.cond_norm/gA))
    #         z = (gpp_pred-dfday2.gpp_norm)#[dfday.VPD > 1]
    #         return z
    #     myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    #     bsres.append(myfit.x)
        
    # bsres = np.array(bsres)
    
    # def tofitLin(pars):
    #     amaxL,kL = pars
        
    #     amax = amaxL*dfday.PPFD_in/(dfday.PPFD_in + kL)

    #     gpp_pred = amax*dfday.cond_norm
    #     z = (gpp_pred-dfday.gpp_norm)#[dfday.VPD > 1]
    #     return z
    # fit0 = np.array([himean,300])
    # myfitLin = scipy.optimize.least_squares(tofitLin,x0=fit0,method="lm",x_scale=np.abs(fit0))

    # amaxL,kL = myfitLin.x
    #%%
    #df["gpp_pred"] = gpp_pred
    #df["kgpp"] = k_pred
    #%%
    gpp2 = np.array(df.gpp)
    for hi in range(24):
        z = gpp2[hi::24]*1
        z2 = np.interp(np.arange(len(z)), np.arange(len(z))[np.isfinite(z)], z[np.isfinite(z)])
        gpp2[hi::24] = z2
    
    df["gpp_interp"] = gpp2
    
    #%%
#    daytime_avg = df.loc[df.NIGHT==0].groupby("date").mean(numeric_only=True).reset_index()
    daytime_avg = df.loc[df.PPFD_in >= 100].groupby("date").mean(numeric_only=True).reset_index()
    #%%
    df["day100"] = df.PPFD_in >= 100
    #%%
    dailydf = df.groupby("date").mean(numeric_only=True).reset_index()
    #%%
    dailydf["cond_daily_dayVPD"] = dailydf.LE/44200/(daytime_avg.VPD/100)

    #%%
    # teff = np.exp(-(daytime_avg.T_AIR-topt1)**2/2/sigma1**2) / np.exp(-(25-topt1)**2/2/sigma1**2)
    # radeff = 1-np.exp(-daytime_avg.PPFD_in/parK)
    # k_pred = gmax*teff*radeff*daytime_avg.LAI
    #%%
    dailydf["cond_daytime"] = daytime_avg.LE/44200/(daytime_avg.VPD/100)
    dailydf["vpd_daytime"] = daytime_avg.VPD
    dailydf["daytime_airt"] = daytime_avg.T_AIR
    dailydf["daytime_par"] = daytime_avg.PPFD_in

    #%%
#    dailydf["dayfrac1"] = 1-dailydf["NIGHT"]
    dailydf["dayfrac1"] = dailydf["day100"]

    #k_pred = daytime_avg.kgpp
    #%%
    #dailydf["gA_daily"] = np.interp(daytime_avg.PPFD_in,parmed,empK) * dailydf.LAI * dailydf.dayfrac1
    #dailydf["amax_daily"] = np.interp(daytime_avg.PPFD_in,parmed,empAmax) * dailydf.LAI * dailydf.dayfrac1
    
    # dailydf["gA_daily"] = gmax1*dailydf["daytime_par"]/(dailydf["daytime_par"] + kG) * dailydf.LAI * dailydf.dayfrac1
    # dailydf["amax_daily"] = amax1*dailydf["daytime_par"]/(dailydf["daytime_par"] + kA) * dailydf.LAI * dailydf.dayfrac1
    # dailydf["gpp_pred_daily"] = dailydf["amax_daily"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_daily"]))
    # #%%
    #dailydf["gpp_pred_hourly2"] = dailydf["amax_hourly"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_hourly"]))

    #%%
    # gpp_pred_daytime = k_pred*(1 - np.exp(-dailydf.cond_daytime/k_pred*slope))
    # gpp_pred_daily = (1-dailydf.NIGHT)*gpp_pred_daytime
    
    #%%
    # dailydf["gA_daytime"] = k_pred/slope
    # dailydf["gpp_slope"] = slope
    #%%
    #%%
    gs2 = pd.merge(daily_gs,dailydf[["date","gpp_interp","daytime_airt","daytime_par","NIGHT","cond_daytime","vpd_daytime"]],on="date",how="left")
#%%
    dfull = gs2.copy()
    dfull = dfull.loc[dfull.rain_qc == 1].copy()
    dfull = dfull.loc[dfull.airt > 0].copy()
    dfull = dfull.loc[dfull.par > 0].copy()
    # dfull["topt"] = topt1
    # dfull["tsigma"] = sigma1
    # dfull["kSWrad"] = parK
    # dfull["gmax0"] = gmax
#%%
   # dfull["rainmax_par_seas"] = np.nanmean(rainymax)
    
    # dfull["amax1"] = amax1
    # dfull["kA"] = kA
    # dfull["gmax1"] = gmax1
    # dfull["kG"] = kG
    #amax1,kA,gmax1,kG
    #dfull["gpp_par_err_rel"] = np.std(bsres[:,0])/amax1
    # dfull["dayfrac"] = 1-dfull.NIGHT
    # dfull["kgpp"] = dfull.gA_daytime*dfull.dayfrac
    # dfull["vpd_fullday"] = 1*dfull.vpd
    # dfull["vpd"] = 1*dfull.vpd_daytime
    #%%
    site_tab.append(dfull)
#%%

site_tab = pd.concat(site_tab).reset_index()
#%%
#site_tab = pd.merge(site_tab,bif_forest[["SITE_ID","LOCATION_LAT","LOCATION_LONG"]],on="SITE_ID",how="left")
#%%
    #x1_norm = (x1-np.min(x1))/(np.max(x1)-np.min(x1))
    #x1_rescale = x1_norm*(np.max(y1)-np.min(y1)) + np.min(y1)
# plt.figure()
# plt.plot(x1-np.nanmean(x1),y1-0.5,'.')
# plt.plot([-200,200],[-0.17,0.17])
# #%%
# plt.plot(x1/521)
# plt.plot(y1)
#%%
site_tab.to_csv("hourly_gs_data_lai75_clim_aug17.csv")
#%%
# segpars = []
# for gmin in [0,200,400,600,800,1000]:
#     dfday2 = dfday.loc[(dfday.PPFD_in >= gmin)*(dfday.PPFD_in < gmin+200)].copy()
#     def tofit(pars):
#         amax,gA = pars
    
#         gpp_pred = amax*(1-np.exp(-dfday2.cond_norm/gA))
#         z = (gpp_pred-dfday2.gpp_norm)#[dfday.VPD > 1]
#         return z
#     fit0 = np.array([himean,himean/150])
#     myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
#     segpars.append(myfit.x)
# segpars = np.array(segpars)
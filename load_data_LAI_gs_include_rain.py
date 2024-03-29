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
import datetime
import matplotlib as mpl
import h5py
#%%
do_bif = 0
if do_bif:
    biftab = pd.read_csv(r"FLX_AA-Flx_BIF_ALL_20200501\FLX_AA-Flx_BIF_HH_20200501.csv")
    groups_to_keep = ["GRP_CLIM_AVG","GRP_HEADER","GRP_IGBP","GRP_LOCATION","GRP_SITE_CHAR","GRP_DOM_DIST_MGMT"]#,"GRP_LAI","GRP_ROOT_DEPTH","SOIL_TEX","SOIL_DEPTH"]
    biftab = biftab.loc[biftab.VARIABLE_GROUP.isin(groups_to_keep)]
    bif2 = biftab.pivot_table(index='SITE_ID',columns="VARIABLE",values="DATAVALUE",aggfunc="first")
    bif2.to_csv("fn2015_bif_tab_h.csv")
#%%
bif_data = pd.read_csv("fn2015_bif_tab.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])]
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH","CRO"])]
#metadata = pd.read_csv(r"C:\Users\nholtzma\Downloads\fluxnet_site_info_all.csv")
#bif_forest = bif_forest.loc[~bif_forest.SITE_ID.isin(["IT-CA1","IT-CA3"])]
all_daily = glob.glob("daily_data\*.csv")
forest_daily = [x for x in all_daily if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
#%%
all_evi = pd.read_csv("evi_ndvi_allsites_sq.csv")

#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%

def cor_skipna2(x,y):
    goodxy = np.isfinite(x*y)
    return np.corrcoef(x[goodxy],y[goodxy])[0,1]

#filename_list = glob.glob("processed_nov7b/*.csv")
#from functions_from_nov16 import prepare_df#, fit_gpp, fit_tau
#from use_w_function import fit_gpp_mm, fit_tau_mm
#from use_w_function_reg import fit_tau_mm_reg

zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf
def getcols(df,varname):
    c1 = [x for x in df.columns if x.startswith(varname+"_") or x==varname]
    return [x for x in c1 if not x.endswith("QC")]

def meancols(df,varname):
    sel_cols = getcols(df,varname)
    if len(sel_cols) == 0:
        return np.nan*np.zeros(len(df))
    col_count = df[sel_cols].count()
    best_col = sel_cols[np.argmax(col_count)]
    return df[best_col]


def lastcols(df,varname):
    sel_cols = getcols(df,varname)
    if len(sel_cols) == 0:
        return np.nan*np.zeros(len(df)), "none"
    best_col = sel_cols[-1]
    return df[best_col], best_col

def fill_na(x):
    return np.interp(np.arange(len(x)), np.arange(len(x))[np.isfinite(x)], x[np.isfinite(x)])

def fill_na2(x,y):
    x2 = 1*x
    x2[np.isnan(x2)] = 1*y[np.isnan(x2)]
    return x2
#%%
#fname = [x for x in forest_daily if site_id in x][0]
def prepare_df(fname, site_id, bif_forest):
    #%%
    df = pd.read_csv(fname,parse_dates=["TIMESTAMP"])
    df[df==-9999] = np.nan
    latdeg = bif_forest.loc[bif_forest.SITE_ID==site_id].LOCATION_LAT.iloc[0]

    #if latdeg < 0:
    #    df["TIMESTAMP"] += datetime.timedelta(days=182)
    
    
    df["date"] = df["TIMESTAMP"].dt.date
    df["hour"] = df["TIMESTAMP"].dt.hour
    df["doy"] = df["TIMESTAMP"].dt.dayofyear
    df["month"] = df["TIMESTAMP"].dt.month

    site_id = fname.split("\\")[-1].split('_')[1]
    #print(site_id)
    
    df["year"] = df["TIMESTAMP"].dt.year
    df["year_new"] = 1*df["year"]
    
    df["doy_new"] = 1*df.doy
    if latdeg < 0:
        df["doy_new"] = (df["TIMESTAMP"] + datetime.timedelta(days=182)).dt.dayofyear
        df["year_new"] = (df["TIMESTAMP"] + datetime.timedelta(days=182)).dt.year

    #%%
    laifile = "lai_csv/lai_csv/" + "_".join(site_id.split("-")) + "_LAI_FLX15.csv"
    try:
        laidf = pd.read_csv(laifile,parse_dates=["Time"])
        df2 = pd.merge(df,laidf,left_on = "TIMESTAMP",right_on="Time",how='left')
        lai_all = np.array(df2.LAI)
        lai_all = np.interp(np.arange(len(lai_all)),
                            np.arange(len(lai_all))[np.isfinite(lai_all)],
                            lai_all[np.isfinite(lai_all)])
        df2 = None
    except:
        lai_all = np.zeros(len(df))
    df["LAI"] = lai_all

    #df["SITE_ID"] = site_id
    #df = pd.merge(df,lai_piv_tab,on=["SITE_ID","month"],how='left')
    #%%
#    laimms = pd.read_csv("lai_2sites_modis.csv")
#    laimms = pd.read_csv("laitest_modis_5k.csv")
#     laimms = pd.read_csv("sqbuff_modis_lai.csv")
# #    laimms = pd.read_csv("modis_test_sq5k.csv")

#     laimms = laimms.loc[laimms.SITE_ID==site_id]
#     laimms["datecode"] = laimms["system:index"].str.slice(stop=10)
#     laimms["date"] = pd.to_datetime(laimms["datecode"],format="%Y_%m_%d").dt.date
    
#     df = pd.merge(df,laimms[["date","Lai"]],on="date",how="left")
#     lai_arr = np.array(df.Lai)
    
#     lai_smooth = np.nan*lai_arr
#     sw = 15
#     for swi in range(sw,len(lai_smooth)-sw):
#         lai_smooth[swi] = np.nanmean(lai_arr[swi-sw:swi+sw+1])
    
#     df["LAI2"] = lai_smooth/10
    #%%
    #laimms = pd.read_csv("evi1km_test.csv")
#    laimms = pd.read_csv("modis_test_sq5k.csv")
#%%
    laimms = all_evi.loc[all_evi.SITE_ID==site_id].copy()
    laimms["datecode"] = laimms["system:index"].str.slice(stop=10)
    laimms["date"] = pd.to_datetime(laimms["datecode"],format="%Y_%m_%d").dt.date
    #%%
    df = pd.merge(df,laimms[["date","EVI"]],on="date",how="left")
    if np.sum(np.isfinite(df.EVI)) == 0:
        df["EVI2"] = np.nan
    
    else:
        lai_arr = np.array(df.EVI)
        
        lai_int = np.interp(np.arange(len(lai_arr)),
                            np.arange(len(lai_arr))[np.isfinite(lai_arr)],
                            lai_arr[np.isfinite(lai_arr)],
                            left=np.nan,right=np.nan)    
        df["EVI2"] = lai_int/10000
    #
    #%%
    # plotLAI = 1
    # if plotLAI:
    #     plt.figure()
    #     plt.plot(df.date[:2000],df.LAI[:2000],'g',label="LAI"); 
    #     plt.xlabel("Date"); plt.ylabel("MODIS LAI");
    #     plt.plot([],[],'r',label="GPP")
    #     plt.twinx()
    #     gpp1 = (df.GPP_DT_VUT_REF + df.GPP_NT_VUT_REF)/2
    #     plt.plot(df.date[:2000],gpp1[:2000],'r')
    #     plt.ylabel(r"GPP $(\mu mol/m^2/s)$");
    #%%
    #par_summer = np.array(meancols(df,'PPFD_IN'))
    #if np.mean(np.isfinite(par_summer)) < 0.5:
    par_summer = np.array(meancols(df,'SW_IN_F'))
    potpar = np.array(meancols(df,'SW_IN_POT'))
    #%%
    airt_summer = np.array(meancols(df,"TA"))
    #rh_summer = np.array(meancols(df,"RH"))/100
    SatVP = 6.1094*np.exp(17.625*airt_summer/ (airt_summer+ 243.04))/10  #kpa
    vpd_summer =  np.array(meancols(df,"VPD"))/10  #SatVP*(1-rh_summer)
    
    #rain_summer = np.array(df['RAIN_daily_mean'])
    # #lai = np.array(df['LAI_modis'])
    
    #et_summer = np.array(df['LE_CORR']) / 44200 
    et_summer = np.array(df['LE_F_MDS']) / 44200 

    et_qc = np.array(df.LE_F_MDS_QC)
    et_summer[et_qc < 0.9] = np.nan
    #%%
    
    le_25 = np.array(df['LE_CORR_25']) #/ 44200 
    le_75 = np.array(df['LE_CORR_75']) #/ 44200 
    #et_summer[np.isnan(le_25*le_75)] = np.nan
    
    myrn = np.array(meancols(df,"NETRAD"))
     
    sw = meancols(df,"SW_IN") -meancols(df,"SW_OUT") 
    lw = meancols(df,"LW_IN") -meancols(df,"LW_OUT") 
    myrn2 = np.array(sw+lw)#-myG
    
    myrn[np.isnan(myrn)] = myrn2[np.isnan(myrn)]
    
    myg = np.array(meancols(df,"G")) #-myG
    if np.mean(np.isfinite(myg)) == 0:
        myg = 0
    #vpd_summer = np.array(df["VPD_F"])/10#*10 #hPa to kPa
    
    vpd_summer[vpd_summer < 0.1] = np.nan
    
    et_summer[et_summer <= 0] = np.nan
    #et_summer[np.isnan(etunc_summer)] = np.nan
    #%%
    try:
        ground_heat = np.array(df["G_F_MDS"])
    #ground_heat[np.isnan(ground_heat)] = 0
    except KeyError:
        ground_heat = 0.1*myrn
    if np.mean(np.isfinite(ground_heat)) < 0.5:
        ground_heat = 0.1*myrn

    rain_summer = np.array(df["P_F"])
    #%%
    if np.sum(np.isfinite(et_summer)) < (25):
        print("Not enough ET")
        return "Not enough data"
        
    #%%
    df['etqc'] = et_summer
    
    my_clim = df.groupby("doy_new").mean(numeric_only=True)
    
    gpp_clim = np.array(1*my_clim["GPP_DT_VUT_REF"] + 1*my_clim["GPP_NT_VUT_REF"])/2
    
    #gpp_clim_std = gpp_clim - np.nanmin(gpp_clim)
    
    gpp_adjoin = fill_na(np.tile(gpp_clim,3))
    
    gpp_clim_smooth = np.zeros(len(gpp_adjoin))
    swidth = 14
    
    for i in range(swidth,len(gpp_adjoin)-swidth):
        gpp_clim_smooth[i] = np.nanmean(gpp_adjoin[i-swidth:i+swidth+1])

    gpp_clim_smooth[:swidth] = np.mean(gpp_clim[:swidth])
    gpp_clim_smooth[-swidth:] = np.mean(gpp_clim[-swidth:])  
    
    gpp_summer = np.array(1*df["GPP_DT_VUT_REF"] + 1*df["GPP_NT_VUT_REF"])/2
    #gpp_summer_nt = np.array(df["GPP_NT_VUT_REF"])

    #airt_summer[airt_summer < 0] = np.nan
    gpp_summer[gpp_summer < 0] = np.nan
    
    nee_qc = np.array(df.NEE_VUT_REF_QC)
    gpp_summer[nee_qc < 0.5] = np.nan
    
    
    
    #gpp_smooth = np.zeros(len(gpp_summer))
    #swidth = 14
    
    #for i in range(swidth,len(gpp_smooth)-swidth):
     #   gpp_smooth[i] = np.nanmean(gpp_summer[i-swidth:i+swidth+1])

    #gpp_smooth[:swidth] = np.mean(gpp_summer[:swidth])
    #gpp_smooth[-swidth:] = np.mean(gpp_summer[-swidth:])
    #%%
    df["gpp_smooth"] = 0 #gpp_smooth
    
#    year95 = df.groupby("year_new").quantile(0.95,numeric_only=True).reset_index()
    year95 = df.groupby("year_new").max(numeric_only=True).reset_index()

    year95["gpp_y95"] = 1*year95["gpp_smooth"]
    year95["lai_y95"] = 1*year95["LAI"]
#%%
#    yearMin = df.groupby("year_new").quantile(0.05,numeric_only=True).reset_index()
    yearMin = df.groupby("year_new").min(numeric_only=True).reset_index()

    yearMin["lai_ymin"] = 1*yearMin["LAI"]
    yearMin["gpp_ymin"] = 1*yearMin["gpp_smooth"]

    #%%
    df = pd.merge(df,year95[["year_new","lai_y95","gpp_y95"]],how="left",on="year_new")
    
    df = pd.merge(df,yearMin[["year_new","lai_ymin","gpp_ymin"]],how="left",on="year_new")
    
    #%%
#    is_summer = df.gpp_smooth/df.gpp_y95 >= 0.5
    #is_summer = df.LAI/df.lai_y95 >= 0.75
    #is_summer_90 = df.LAI/df.lai_y95 >= 0.9
#    df["LAI_gt50"] = (df.LAI-df.lai_ymin)/(df.lai_y95-df.lai_ymin) > 0.67
    df["LAI_gt50"] = (df.LAI-0)/(df.lai_y95-0) > 0.8

    #df["LAI_gt50"] = (df.gpp_smooth/df.gpp_y95) > 0.67
    year_list = pd.unique(df.year_new)
    gs_starts = []
    gs_ends = []
    for year in year_list:
        dfy = df.loc[df.year_new==year].reset_index()
        topday = np.argmax(dfy.LAI)
#        topday = np.argmax(dfy.gpp_smooth)

        under50 = np.where(~dfy.LAI_gt50)[0]
        try:
            summer_start = under50[under50 < topday][-1]
        except:
            summer_start = 0
        try:
            summer_end = under50[under50 > topday][0]
        except:
            summer_end = 365
        gs_starts.append(summer_start)
        gs_ends.append(summer_end)
        
    summer_df = pd.DataFrame({"year_new":year_list,
                              "summer_start":gs_starts,
                              "summer_end":gs_ends})
    df= pd.merge(df,summer_df,on="year_new",how="left")
    is_summer = np.array((df.doy_new >= df.summer_start)*(df.doy_new <= df.summer_end))
   # is_summer = (df.gpp_smooth-df.gpp_ymin)/(df.gpp_y95-df.gpp_ymin) > 0.50

#%%
#     gpp_clim_smooth_raw = gpp_clim_smooth[366:366*2]
#     gpp_clim_smooth = gpp_clim_smooth_raw #- np.min(gpp_clim_smooth_raw)
#     topday = np.argmax(gpp_clim_smooth)
#     under50 = np.where(gpp_clim_smooth < 0.67*np.nanmax(gpp_clim_smooth))[0]
# #%%
#     try:
#         summer_start = under50[under50 < topday][-1]
#     except IndexError:
#         summer_start = np.where(np.isfinite(gpp_clim_smooth))[0][0]
#     try:
#         summer_end = under50[under50 > topday][0]
#     except IndexError:
#         summer_end = np.where(np.isfinite(gpp_clim_smooth))[0][-1]
#     #topday = summer_start
    #%%
    plot_gs = 0
    if plot_gs:
        #%%
        plt.figure()
        plt.plot(gpp_clim)
        plt.plot(gpp_clim_smooth)
        #plt.axvspan(summer_start,summer_end,color="green",alpha=0.33)
        plt.twinx()
        plt.plot(my_clim.SW_IN_POT,'r')
        plt.twinx()
        plt.plot(my_clim.LAIclim,'k')

    #%%
    #p_in_clim = fill_na(np.array(my_clim.P_F))
#    et_out_clim =fill_na(np.array(my_clim["LE_F_MDS"] / 44200 * 18/1000 * 60*60*24))
    #et_out_clim =fill_na(np.array(my_clim["etqc"] * 18/1000 * 60*60*24))
    
    #turn_point = np.argmax(np.cumsum(p_in_clim - et_out_clim) )
    
    my_clim = my_clim.reset_index()
    my_clim["P_F_c"] = fill_na(np.array(my_clim.P_F))
#    my_clim["LE_all_c"] = fill_na(np.array(my_clim.LE_F_MDS))
    my_clim["LE_all_c"] = fill_na(np.array(my_clim.etqc))

    dfm = pd.merge(df,my_clim[["doy_new","P_F_c","LE_all_c"]],on="doy_new",how="left")        
    
    p_in = fill_na2(np.array(df.P_F),np.array(dfm.P_F_c))
    et_out = fill_na2(et_summer * 18/1000 * 60*60*24,np.array(dfm["LE_all_c"] * 18/1000 * 60*60*24))
    doy_summer = np.array(df["doy_new"])
    #%%
    df["et_wqc"] = et_summer
    #yearcount = df.groupby("year").count().reset_index()

    # if np.mean(et_out) < np.mean(p_in):
    #     #inflow = 0
    #     yeardf = df.groupby("year_new").sum(numeric_only=True).reset_index()
    #     yearET = yeardf.LE_F_MDS / 44200 * (18/1000) * (60*60*24)
    #     bad_year = yeardf.loc[yeardf.P_F < yearET].year_new
        
    #     to_replace = df.year_new.isin(bad_year)
    #     p_in[to_replace] = dfm.P_F_c[to_replace]
    #     et_out[to_replace] = dfm.LE_all_c[to_replace] * 18/1000 * 60*60*24
    
    # else:
    # #     inflow = np.mean(et_out) - np.mean(p_in)
    #       to_replace = []
    #    # return "ET exceeds P"
    #%%
    # cbal = np.cumsum(p_in - et_out)
    # cdiff = 0*cbal
    # cdiff[182:-182] = (cbal[364:] - cbal[:-364])/364
    
    #%%
    inflow = max(0, np.mean(et_out) - np.mean(p_in))
    #inflow = np.mean(et_out) - np.mean(p_in)
    #pfrac = min(1,np.mean(et_out)/np.mean(p_in))
    
    wbi = 0
    waterbal_raw = np.zeros(len(doy_summer))
    for dayi in range(len(p_in)):
        #if doy_summer[dayi]==1:
        #    wbi=0
        waterbal_raw[dayi] = wbi
        wbi += p_in[dayi] - et_out[dayi] #- cdiff[dayi]
        wbi = min(0,wbi)
        #wbi -= 0.01*max(0,wbi)
        #if dayi == opposite_peak:
        #    wbi = 0
    waterbal_corr = 1*waterbal_raw
    #waterbal_corr[to_replace] = np.nan
    #%%
    # x1 = 0
    # x2 = 0
    
    # waterbal_raw = np.zeros((len(doy_summer),2))
    # #%%
    # for dayi in range(len(p_in)):
    #     waterbal_raw[dayi,:] = [x1,x2]
    #     qi = 0.01*(x1-x2)
    #     #runoff = 0.01*x1
    #     x1 += p_in[dayi]  -qi#- runoff #- qi
    #     x2 += qi - et_out[dayi] 
    #     x1 = min(0,x1)
    #     #wbi -= 0.01*max(0,wbi)
    #     #if dayi == opposite_peak:
    #     #    wbi = 0
    # waterbal_corr = 1*waterbal_raw
    
    #%%
    #waterbal_corr[np.isnan(et_summer)] = np.nan
    #%%
    # doynew_arr = np.array(df.doy_new)
    # #year_arr = np.array(df.year)

    # mcount = 0
    # for di in range(len(df)):
    #     if doynew_arr[di] == 1:
    #         mcount = 0
    #     if np.isnan(waterbal_corr[di]):
    #         mcount += 1
    #     if mcount > 60:
    #         waterbal_corr[di] = np.nan
    #%%
    df["waterbal"] = waterbal_corr
    #%%
    # def avgpast(x,w):
    #     y = 1*x
    #     for i in range(w,len(x)):
    #         y[i] = np.mean(x[i-w:i])
    #     return y
    def avgpast2(x,w):
        y = np.zeros(len(x))
        for i in range(w):
            y[w:] += x[(w-i):(len(x)-i)]
        y /= w
        y[y==0] = np.nan
        return y
    #%%
    
    smc_summer, smc_name = lastcols(df,'SWC')
    smc_summer = np.array(smc_summer)
    
    try:
        smc_qc = np.array(df[smc_name + "_QC"])
        smc_summer[smc_qc==0] = np.nan
        bothgood = np.isfinite(smc_summer*waterbal_corr)
        # try:
        #     sinterp = np.interp(smc_summer,np.sort(smc_summer[bothgood]),np.sort(waterbal_corr[bothgood]))
        # except ValueError:
        #     sinterp = np.nan*smc_summer
    except KeyError:
        smc_summer = np.nan*waterbal_corr
        #sinterp = np.nan*waterbal_corr

    df["smc"] = smc_summer
    #%%
    # df_yearmean = df.groupby("year").mean(numeric_only=True).reset_index()
    # df3 = pd.merge(df,df_yearmean[["year","waterbal","smc"]],on='year',how='left')
    # wb_anom = np.array(df3.waterbal_x - df3.waterbal_y)
    # s_anom = np.array(df3.smc_x - df3.smc_y)
    # bothfin = np.isfinite(s_anom*wb_anom)
    # try:
    #     sinterp_anom = np.interp(s_anom,np.sort(s_anom[bothfin]),np.sort(wb_anom[bothfin]))
    #     sinterp_full = sinterp_anom + np.array(df3.smc_y)*np.nanstd(wb_anom)/np.nanstd(s_anom)
    # except ValueError:
    #     sinterp_full = np.nan*waterbal_corr
    #%%
    
    #%%
    #df["smc_lag1"] = adj_smc0
    #df2 = df.loc[df.doy_new != 366]
    #my_clim = df2.groupby("doy_new").mean(numeric_only=True)
    
    # smctile = np.tile(my_clim.smc,3)
    # wbtile = np.tile(my_clim.waterbal,3)
    
    # lag_cors = []
    # for lag in range(1,180):
    #     #print(lag)
    #     lag_cors.append(cor_skipna2(avgpast2(smctile,lag)[366:(366*2)],wbtile[366:(366*2)]))
    # #
    
    #%%
    # lag_cors = []
    # for lag in range(1,180):
    #     #print(lag)
    #     lag_cors.append(cor_skipna2(avgpast2(smc_summer,lag),waterbal_corr))
    # #%%
    # best_lag = range(1,180)[np.argmax(lag_cors)]
    # adj_smc0 = avgpast2(smc_summer,best_lag)
    # #%%
    # #adj_smc *= np.nanstd(waterbal_corr[np.isfinite(adj_smc)])/np.nanstd(adj_smc[np.isfinite(adj_smc)])
    # df["smc_lag1"] = adj_smc0
    # df2 = df.loc[df.doy_new != 366]
    # my_clim = df2.groupby("doy_new").mean(numeric_only=True)
    # adj_smc = adj_smc0 * np.nanstd(my_clim.waterbal)/np.nanstd(my_clim.smc_lag1)
    # # lagmat = np.zeros((len(waterbal_corr),50))
    # for z in range(50):
    #     lagmat[:,z] = np.interp(np.arange(len(waterbal_corr))+z,
    #                             np.arange(len(waterbal_corr))[np.isfinite(smc_summer)],
    #                             smc_summer[np.isfinite(smc_summer)])
    #     #%%
    # lagreg = sm.OLS(waterbal_corr, sm.add_constant(lagmat)).fit()
    
    #%%
    
    #then multiply by std ratio 
    #%%
    
   # plt.plot(avgpast(smc_summer,80),waterbal_corr,'.')
    #ymin = df.groupby("year").min().reset_index().rename(columns={"waterbal":"wb_ymin"})
    #ymax = df.groupby("year").max().reset_index().rename(columns={"waterbal":"wb_ymin"})
#%%
    # smc_smooth = np.nan*smc_summer
    # for x in range(182,len(smc_smooth)-182):
    #     z = smc_summer[x-182:x+183]
    #     if np.mean(np.isfinite(z)) > 0.75:
    #         smc_smooth[x] = np.nanmean(smc_summer[x-182:x+183])
    # #ymin
    # #%%
    # wb_smooth = np.nan*smc_summer
    # for x in range(182,len(smc_smooth)-182):
    #     z = waterbal_corr[x-182:x+183]
    #     if np.mean(np.isfinite(z)) > 0.75:
    #         smc_smooth[x] = np.nanmean(smc_summer[x-182:x+183])
    #%%
    #is_summer = (doy_summer >= summer_start)*(doy_summer <= summer_end)
    #is_late_summer = (doy_summer >= topday)*(doy_summer <= summer_end)
    #%%
    
    
   # ground_heat = 0
    
    SatVP = 6.1094*np.exp(17.625*airt_summer/ (airt_summer+ 243.04))/10  #kpa
    
    wsarr = np.array(meancols(df,'WS'))
    
    #wsarr[wsarr == 0] = 0.025
    # myga_old = 0.41**2*wsarr / (np.log(2.4/35))**2
    ustar = np.array(meancols(df,"USTAR"))
    #myga = (wsarr/ustar**2 + 6.2*ustar**(-2/3))**-1
    myga = ustar**2/wsarr
    
    lambda0 = 2.26*10**6
    sV = 0.04145*np.exp(0.06088*airt_summer) #in kpa
    gammaV = 100*1005/(lambda0*0.622) #in kpa
    
    petVnum = (sV*(myrn-ground_heat) + 1.225*1000*vpd_summer*myga)*(myrn > 0) #/(sV+gammaV*(1+myga[i]/(gmax*condS*mylai[i])))  #kg/s/m2 
    
    g_ratio = (petVnum / (et_summer*44200) - sV)/gammaV - 1
    inv2 = myga/g_ratio
    
    
    inv2_stp = inv2/0.0224
    
    patm_summer =  np.array(meancols(df,"PA"))
    patm_summer[np.isnan(patm_summer)] = 101.325
    
    gasvol_fac = (airt_summer + 273.15)/(25+273.15) * 101.325/patm_summer
    
    inv2_varTP = inv2/(22.4*gasvol_fac/1000)
    
    daily_cond = inv2_varTP
    daily_cond[daily_cond > 2] = np.nan
    daily_cond[daily_cond <= 0] = np.nan
    
    
    pet = petVnum/(sV+gammaV)

    #%%
    if np.sum(np.isfinite(gpp_summer)) < (25):
        print("Not enough GPP")
        return "Not enough data"

    #norm_cond = daily_cond/lai_summer
    #norm_gpp = gpp_summer/lai_summer
    #norm_gpp[norm_gpp > 8] = np.nan
    #%%
    houri = 12
    deg_noon = 360 / 365 * (doy_summer + houri / 24 + 10);
    decd = -23.44*np.cos(deg_noon*np.pi/180)
    lhad = (houri-12)*15
    
    cosz = (np.sin(latdeg*np.pi/180) * np.sin(decd*np.pi/180) + 
            np.cos(latdeg*np.pi/180) * np.cos(decd*np.pi/180) *
            np.cos(lhad*np.pi/180))
      
    #%%
    petVnum[petVnum==0] = np.nan
    gpp_summer = np.array(gpp_summer)
    
    rain_prev = 0*rain_summer
    rain_prev[1:] = rain_summer[:-1]
    #%%
    rain_fake = 1*rain_summer
    #rain_fake[doy_summer==summer_end] = np.inf
    rain_for_dict = [rain_fake[is_summer],np.array(df.year)[is_summer]]
    
    # rain_fake = 1*rain_summer
    # rain_fake[doy_summer==365] = np.inf
    # rain_for_dict = [rain_fake,np.array(df.year)]
#%%
    df_to_fit_full = pd.DataFrame({"date":df.date,"airt":airt_summer,"year":df.year,"year_new":df.year_new,
                              "par":par_summer,#"cosz":cosz,
                              "potpar":potpar,
                              #"potpar_mean":np.nanmean(potpar),
                              #"potpar_max":np.nanmax(potpar),
                              #"potpar_min":np.nanmin(potpar),
                              #"smc_lag":adj_smc,
                              "cond":daily_cond,"gpp":gpp_summer,
                              "doy":doy_summer,"vpd":vpd_summer,
                              "doy_raw":np.array(df.doy),
                              "waterbal":waterbal_corr,
                              "ET":et_summer,"ET_qc":et_qc,
                              #"gasvol_fac":gasvol_fac,
                              #"petVnum":petVnum,
                              #"myga":myga,"sV":sV,
                              "rain":rain_summer,
                              "rain_prev":rain_prev,
                              "LAI":lai_all,
                              #"lai_yq95":df.lai_y95,
                              "smc":smc_summer,
                              #"et_unc":df.LE_RANDUNC/44200,
                              #"sinterp":sinterp,
                              #"sinterp_anom":sinterp_full,
                              #"nee_unc":df.NEE_VUT_REF_RANDUNC,#,/-df.NEE_VUT_REF,
                              #"gpp_unc_DT":(df.GPP_DT_VUT_75-df.GPP_DT_VUT_25),#/df.GPP_DT_VUT_REF,
                              #"gpp_unc_NT":(df.GPP_NT_VUT_75-df.GPP_NT_VUT_25),#/df.GPP_NT_VUT_REF,
                              #"gpp_unc":(df.GPP_DT_VUT_75-df.GPP_DT_VUT_25)/df.GPP_DT_VUT_REF,
                              #"gpp_smooth":gpp_smooth,
                              #"gpp_yq95":df.gpp_y95,
                              #"gpp_nt" : gpp_summer_nt,
                              #"summer_start":summer_start,

                              #"summer_end":summer_end,
                              #"summer_peak":topday,
                              "EVI2":df.EVI2,
                              "is_summer":is_summer,
                              "rain_qc":df.P_F_QC
                              #"smc_std_all" : np.nanstd(smc_summer),
                              #"wbal_std_all" : np.nanstd(waterbal_corr),
                              #"smc_mean_all" : np.nanmean(smc_summer),
                              #"wbal_mean_all" : np.nanmean(waterbal_corr),
                              #"PET":pet
                              })
    
    def fill_summer(x):
        ans = np.zeros(len(df))
        ans[is_summer] = 1*x
        return ans
    #df_to_fit_full["smc_iav_ratio"] = np.nanstd(df3.smc_y)/np.nanstd(df3.smc_x)
    #df_to_fit_full["wb_iav_ratio"] = np.nanstd(df3.waterbal_y)/np.nanstd(df3.waterbal_x)

#    df_to_fit = df_to_fit_full.loc[is_summer]#.dropna(subset = set(df_to_fit_full.columns)-{"smc","sinterp","sinterp_anom","smc_lag","EVI2"})
#    df_to_fit = df_to_fit_full.loc[is_summer].reset_index()#.dropna(subset = set(df_to_fit_full.columns)-{"smc","sinterp","sinterp_anom","smc_lag","EVI2"})
    df_to_fit = df_to_fit_full.reset_index()#.dropna(subset = set(df_to_fit_full.columns)-{"smc","sinterp","sinterp_anom","smc_lag","EVI2"})

    #df_to_fit = df_to_fit.loc[df_to_fit.par >= 100]
    
    #df_to_fit = df_to_fit.loc[df_to_fit.rain==0]
    #df_to_fit = df_to_fit.loc[df_to_fit.rain_prev==0]

    
    df_to_fit["inflow"] = inflow

#    df_to_fit = df_to_fit.loc[(df_to_fit.doy >= topday)*(df_to_fit.vpd >= 0.5)].copy()
#%%
#    df_to_fit = df_to_fit.loc[df_to_fit.doy >= topday].copy()
    #df_to_fit = df_to_fit.loc[(df_to_fit.et_unc / df_to_fit.ET) <= 0.2].copy()
    #df_to_fit["gpp_unc"] = (df_to_fit["gpp_unc_DT"] + df_to_fit["gpp_unc_NT"])/2/df_to_fit.gpp
    
    #%%
    #gpp_qc = np.array(df_to_fit.gpp)
    #gpp_qc[gpp_qc <= 0] = np.nan
    #gpp_qc[(df_to_fit.gpp_unc <= 0) | (df_to_fit.gpp_unc > 0.25)] = np.nan
    #df_to_fit = df_to_fit.loc[(df_to_fit.gpp_unc >= 0)*(df_to_fit.gpp_unc < 0.15)].copy()
    #df_to_fit["gpp_qc"] = gpp_qc
    
    #df_to_fit = df_to_fit.loc[df_to_fit.rain_prev==0]
    #df_to_fit = df_to_fit.loc[df_to_fit.vpd >= 0.5]
#%%
    #df_to_fit = df_to_fit.loc[df_to_fit.vpd > 0.5].copy()
    #df_to_fit = df_to_fit.loc[df_to_fit.waterbal > -500].copy()


#%%
    #year_count = df_to_fit.groupby("year").count()
    #year_count["ETcount"] = year_count["ET"]*1
    #year_count["gpp_count"] = year_count["gpp_qc"]*1

    #df_to_fit = pd.merge(df_to_fit,year_count[["ETcount","gpp_count"]],on="year",how='left')
    #df_to_fit = df_to_fit.loc[df_to_fit["ETcount"] >= 10].copy()
    #df_to_fit = df_to_fit.loc[df_to_fit["gpp_count"] >= 10].copy()
    
    # df_to_fit["mat_data"] = np.nanmean(airt_summer)
    # df_to_fit["map_data"] = np.nanmean(p_in)
    # df_to_fit["mgsp_data"] = np.nanmean(p_in[is_summer])
    # df_to_fit["mean_netrad"] = np.nanmean(myrn)
    # df_to_fit["gs_netrad"] = np.nanmean(myrn[is_summer])
    df_to_fit["SITE_ID"] = site_id
#%%

    return df_to_fit, rain_for_dict, np.nanmean(airt_summer), np.nanmean(rain_summer), df_to_fit_full
#%%
#%%
#%%
#width = 1
zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf
#%%

rain_dict = {}
year_tau_dict = {}
all_results = []
site_result = {}
#%%
good_sites = ['AU-Ade', 'AU-Cpr', 'AU-Dry', 'AU-Emr', 'AU-Gin', 'AU-Stp',
       'AU-TTE', 'CA-Oas', 'CA-TP4', 'CN-Du2', 'ES-LJu', 'IT-Noe',
       'NL-Loo', 'SD-Dem', 'US-AR1', 'US-AR2', 'US-Blo', 'US-Cop',
       'US-Me2', 'US-Me3', 'US-Me5', 'US-Me6', 'US-NR1', 'US-SRC',
       'US-SRG', 'US-SRM', 'US-UMB', 'US-UMd', 'US-Var', 'US-Whs',
       'US-Wkg']

#%%
for fname in forest_daily:#[forest_daily[x] for x in [70,76]]:
#%%
    site_id = fname.split("\\")[-1].split('_')[1]
    #if site_id not in good_sites:
    #    continue
    #if not site_id.startswith("US"):
    #     continue
    #if site_id.startswith("AU") or site_id.startswith("US"):
    #    continue
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
        site_result[site_id] = "Not enough data"

        continue
    #%%
    # if sum(np.isfinite(df_to_fit.gpp_qc)) < 25:
    #     print("Not enough data")
    #     site_result[site_id] = "Not enough data"
    #     continue
    #%%

    #%%
    #dfi = df_to_fit.loc[df_to_fit.waterbal < np.nanmedian(df_to_fit.waterbal)].copy()

   # dfi = df_to_fit.loc[(df_to_fit.doy >= topday)].copy()

    #%%    
    all_results.append(df_to_fit)
    #%%
all_results = pd.concat(all_results)
#all_results.to_csv("gs_80lai_max_april10.csv")
all_results.to_csv("all_yearsites.csv")

#%%
# sites = []
# years = []
# rains = []
# for x in rain_dict.keys():
#     ri = rain_dict[x][0]
#     sites.append(np.array([x]*len(ri)))
#     years.append(rain_dict[x][1])
#     rains.append(ri)
# #%%
# raindf = pd.DataFrame({"SITE_ID":np.concatenate(sites),
#                       "year":np.concatenate(years),
#                       "rain_mm":np.concatenate(rains)})
# raindf.to_csv("rain_80lai_max_april10.csv")
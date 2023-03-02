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
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])]
#metadata = pd.read_csv(r"C:\Users\nholtzma\Downloads\fluxnet_site_info_all.csv")
bif_forest = bif_forest.loc[~bif_forest.SITE_ID.isin(["IT-CA1","IT-CA3"])]
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
lai_map = h5py.File("LAI_mean_monthly_1981-2015.nc4","r")
#%%
lai_max = np.max(lai_map["LAI"],axis=0)
lai_max[lai_max < 0] = np.nan
#%%

site_ilat = np.floor((np.array(bif_forest.LOCATION_LAT)+90)*4).astype(int)
site_ilon = np.floor((np.array(bif_forest.LOCATION_LONG)+180)*4).astype(int)
#%%
plt.figure()
plt.imshow(lai_max[-1::-1,:]);
plt.plot(site_ilon,720-site_ilat,'r.')
#%%
lai_allmo = np.array(lai_map["LAI"])[:,site_ilat,site_ilon]
lai_allmo[lai_allmo < 0] = np.nan
#%%
month_template = [8,9,10,11,12,1,2,3,4,5,6,7]
lai_pivot = lai_allmo.T.reshape(-1,1)[:,0]
lai_piv_tab = pd.DataFrame({"SITE_ID":np.repeat(bif_forest.SITE_ID,12),
                            "LAIclim":lai_pivot,
                            "month":np.tile(month_template,len(bif_forest))})
#%%
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
    
    df["doy_new"] = 1*df.doy
    if latdeg < 0:
        df["doy_new"] = (df["TIMESTAMP"] + datetime.timedelta(days=182)).dt.dayofyear
    #%%
    #df["SITE_ID"] = site_id
    #df = pd.merge(df,lai_piv_tab,on=["SITE_ID","month"],how='left')
    
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
    et_summer[et_qc < 0.5] = np.nan
    
    
    le_25 = np.array(df['LE_CORR_25']) #/ 44200 
    le_75 = np.array(df['LE_CORR_75']) #/ 44200 
    et_summer[np.isnan(le_25*le_75)] = np.nan
    
    
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
    
    
    gpp_clim_std = gpp_clim - np.nanmin(gpp_clim)
    
    gpp_adjoin = fill_na(np.tile(gpp_clim,3))
    
    
    gpp_smooth = np.zeros(len(gpp_adjoin))
    for i in range(15,len(gpp_smooth)-15):
        gpp_smooth[i] = np.nanmean(gpp_adjoin[i-15:i+16])
    
    gpp_clim_smooth_raw = gpp_smooth[366:366*2]
    gpp_clim_smooth = gpp_clim_smooth_raw #- np.min(gpp_clim_smooth_raw)
    topday = np.argmax(gpp_clim_smooth)
    under50 = np.where(gpp_clim_smooth < 0.5*np.nanmax(gpp_clim_smooth))[0]

    try:
        summer_start = under50[under50 < topday][-1]
    except IndexError:
        summer_start = np.where(np.isfinite(gpp_clim))[0][0]
    try:
        summer_end = under50[under50 > topday][0]
    except IndexError:
        summer_end = np.where(np.isfinite(gpp_clim))[0][-1]
    #topday = summer_start
    #%%
    plot_gs = 0
    if plot_gs:
        #%%
        plt.figure()
        plt.plot(gpp_clim)
        plt.plot(gpp_clim_smooth)
        plt.axvspan(summer_start,summer_end,color="green",alpha=0.33)
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
    #df["et_wqc"] = et_summer
    #yearcount = df.groupby("year").count().reset_index()

    #if np.mean(et_out) < np.mean(p_in):
        #inflow = 0
        # yeardf = df.groupby("year").sum(numeric_only=True).reset_index()
        # yearET = yeardf.LE_F_MDS / 44200 * (18/1000) * (60*60*24)
        # bad_year = yeardf.loc[yeardf.P_F < yearET].year
        
        # to_replace = df.year.isin(bad_year)
        # p_in[to_replace] = dfm.P_F_c[to_replace]
        # et_out[to_replace] = dfm.LE_all_c[to_replace] / 44200 * 18/1000 * 60*60*24
    
    # else:
    #     inflow = np.mean(et_out) - np.mean(p_in)
    #     to_replace = []
        #return "ET exceeds P"
    #%%
    
    inflow = max(0, np.mean(et_out) - np.mean(p_in))
    
    wbi = 0
    waterbal_raw = np.zeros(len(doy_summer))
    for dayi in range(len(p_in)):
        waterbal_raw[dayi] = wbi
        wbi += p_in[dayi] - et_out[dayi] #- 1.8
        wbi = min(0,wbi)
        #if dayi == opposite_peak:
        #    wbi = 0
    waterbal_corr = 1*waterbal_raw
    #waterbal_corr[to_replace] = np.nan
    
    #%%
    waterbal_corr[np.isnan(et_summer)] = np.nan
    #%%
    doynew_arr = np.array(df.doy_new)
    #year_arr = np.array(df.year)

    mcount = 0
    for di in range(len(df)):
        if doynew_arr[di] == 1:
            mcount = 0
        if np.isnan(waterbal_corr[di]):
            mcount += 1
        if mcount > 60:
            waterbal_corr[di] = np.nan
    #%%
    df["waterbal"] = waterbal_corr
    #ymin = df.groupby("year").min().reset_index().rename(columns={"waterbal":"wb_ymin"})
    #ymax = df.groupby("year").max().reset_index().rename(columns={"waterbal":"wb_ymin"})

    #ymin
    #%%
    is_summer = (doy_summer >= summer_start)*(doy_summer <= summer_end)
    is_late_summer = is_summer #(doy_summer >= topday)*(doy_summer <= summer_end)
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
    gpp_summer = np.array(2*df["GPP_DT_VUT_REF"] + 0*df["GPP_NT_VUT_REF"])/2
    gpp_summer_nt = np.array(df["GPP_NT_VUT_REF"])


    #airt_summer[airt_summer < 0] = np.nan
    gpp_summer[gpp_summer < 0] = np.nan
    
    nee_qc = np.array(df.NEE_VUT_REF_QC)
    gpp_summer[nee_qc < 0.5] = np.nan
    
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
    rain_fake[doy_summer==summer_end] = np.inf
    rain_for_dict = [rain_fake[is_late_summer],np.array(df.year)[is_late_summer]]
    #not sure whether using rain over entire or late summer is more appropriate
#    rain_dict[site_id] = [rain_fake[is_summer],np.array(df.year)[is_summer]]

    
    #%%
    df_to_fit_full = pd.DataFrame({"date":df.date,"airt":airt_summer,"year":df.year,
                              "par":par_summer,"cosz":cosz,
                              "potpar":potpar,
                              "potpar_mean":np.nanmean(potpar),
                              "potpar_max":np.nanmax(potpar),
                              "potpar_min":np.nanmin(potpar),
                              "cond":daily_cond,"gpp":gpp_summer,
                              "doy":doy_summer,"vpd":vpd_summer,
                              "doy_raw":np.array(df.doy),
                              "waterbal":waterbal_corr,
                              "ET":et_summer,
                              "gasvol_fac":gasvol_fac,
                              "petVnum":petVnum,
                              "myga":myga,"sV":sV,
                              "rain":rain_summer,
                              "rain_prev":rain_prev,
                              #"smc":smc_summer,
                              "vpd":vpd_summer,
                              "et_unc":df.LE_RANDUNC/44200,
                              #"sinterp":sinterp
                              "nee_unc":df.NEE_VUT_REF_RANDUNC,#,/-df.NEE_VUT_REF,
                              #"gpp_unc_DT":(df.GPP_DT_VUT_75-df.GPP_DT_VUT_25),#/df.GPP_DT_VUT_REF,
                              #"gpp_unc_NT":(df.GPP_NT_VUT_75-df.GPP_NT_VUT_25),#/df.GPP_NT_VUT_REF,
                              "gpp_unc":(df.GPP_DT_VUT_75-df.GPP_DT_VUT_25)/df.GPP_DT_VUT_REF,
                              "gpp_nt" : gpp_summer_nt,
                              "summer_start":summer_start,

                              "summer_end":summer_end,
                              "summer_peak":topday,
                              #"PET":pet
                              })
    
    def fill_summer(x):
        ans = np.zeros(len(df))
        ans[is_summer] = 1*x
        return ans
    
    df_to_fit = df_to_fit_full.loc[is_summer].dropna(subset = set(df_to_fit_full.columns)-{"sinterp_anom","sint_mult","sinterp_mean2","sinterp_anom2"})

    df_to_fit = df_to_fit.loc[df_to_fit.par >= 100]
    
    df_to_fit = df_to_fit.loc[df_to_fit.rain==0]
    #df_to_fit = df_to_fit.loc[df_to_fit.rain_prev==0]

    
    df_to_fit["inflow"] = inflow

#    df_to_fit = df_to_fit.loc[(df_to_fit.doy >= topday)*(df_to_fit.vpd >= 0.5)].copy()
#%%
    #df_to_fit = df_to_fit.loc[df_to_fit.doy >= topday].copy()
    #df_to_fit = df_to_fit.loc[(df_to_fit.et_unc / df_to_fit.ET) <= 0.2].copy()
    #df_to_fit["gpp_unc"] = (df_to_fit["gpp_unc_DT"] + df_to_fit["gpp_unc_NT"])/2/df_to_fit.gpp
    
    #%%
    gpp_qc = np.array(df_to_fit.gpp)
    gpp_qc[gpp_qc <= 0] = np.nan
    gpp_qc[(df_to_fit.gpp_unc <= 0) | (df_to_fit.gpp_unc > 0.25)] = np.nan
    #df_to_fit = df_to_fit.loc[(df_to_fit.gpp_unc >= 0)*(df_to_fit.gpp_unc < 0.15)].copy()
    df_to_fit["gpp_qc"] = gpp_qc
    
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
    
    df_to_fit["mat_data"] = np.nanmean(airt_summer)
    df_to_fit["map_data"] = np.nanmean(p_in)
    df_to_fit["mgsp_data"] = np.nanmean(p_in[is_summer])
    df_to_fit["mean_netrad"] = np.nanmean(myrn)
    df_to_fit["gs_netrad"] = np.nanmean(myrn[is_summer])
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
        site_result[site_id] = "Not enough data"

        continue
    #%%
    if sum(np.isfinite(df_to_fit.gpp_qc)) < 25:
        print("Not enough data")
        site_result[site_id] = "Not enough data"
        continue
    #%%

    #%%
    #dfi = df_to_fit.loc[df_to_fit.waterbal < np.nanmedian(df_to_fit.waterbal)].copy()

   # dfi = df_to_fit.loc[(df_to_fit.doy >= topday)].copy()

    #%%    
    all_results.append(df_to_fit)
    #%%
all_results = pd.concat(all_results)
all_results.to_csv("gs_50_50_mar1b.csv")
#%%
sites = []
years = []
rains = []
for x in rain_dict.keys():
    ri = rain_dict[x][0]
    sites.append(np.array([x]*len(ri)))
    years.append(rain_dict[x][1])
    rains.append(ri)
#%%
raindf = pd.DataFrame({"SITE_ID":np.concatenate(sites),
                      "year":np.concatenate(years),
                      "rain_mm":np.concatenate(rains)})
raindf.to_csv("rain_50_50_mar1b.csv")
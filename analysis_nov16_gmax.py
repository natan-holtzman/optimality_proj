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

#%%
#%%
#width = 1
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

def fill_na(x):
    return np.interp(np.arange(len(x)), np.arange(len(x))[np.isfinite(x)], x[np.isfinite(x)])

def fill_na2(x,y):
    x2 = 1*x
    x2[np.isnan(x2)] = 1*y[np.isnan(x2)]
    return x2

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
    df = pd.read_csv(fname,parse_dates=["TIMESTAMP"])
    df[df==-9999] = np.nan
    
    df["date"] = df["TIMESTAMP"].dt.date
    df["hour"] = df["TIMESTAMP"].dt.hour
    df["doy"] = df["TIMESTAMP"].dt.dayofyear
    site_id = fname.split("\\")[-1].split('_')[1]
    print(site_id)
    latdeg = bif_forest.loc[bif_forest.SITE_ID==site_id].LOCATION_LAT.iloc[0]
    if latdeg < 0:
        df["doy"] = (df["doy"]+182) % 365
    df["year"] = df["TIMESTAMP"].dt.year
    #%%
    
    par_summer = np.array(meancols(df,'PPFD_IN'))
    if np.mean(np.isfinite(par_summer)) < 0.5:
        par_summer = np.array(meancols(df,'SW_IN_F'))*2
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
    
    #%%
    et_summer[et_summer <= 0] = np.nan
    #et_summer[np.isnan(etunc_summer)] = np.nan
    #%%
    # plt.figure()
    # plt.plot(vpd_summer,et_summer,".")
    # plt.xlim(0,5)
    # plt.ylim(0,0.006)
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
        site_result[site_id] = "Not enough data"
        continue
        
    #%%
    my_clim = df.groupby("doy").mean(numeric_only=True)
    
    gpp_clim = np.array(my_clim["GPP_NT_VUT_REF"])
    
    
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
    #%%
    # plt.figure()
    # plt.plot(gpp_clim)
    # plt.plot(gpp_clim_smooth)
    # plt.axvspan(summer_start,summer_end,color="green",alpha=0.33)
    #%%
    
    #%%
    p_in_clim = fill_na(np.array(my_clim.P_F))
    et_out_clim =fill_na(np.array(my_clim["LE_F_MDS"] / 44200 * 18/1000 * 60*60*24))
    
    turn_point = np.argmax(np.cumsum(p_in_clim - et_out_clim) )
    
    my_clim = my_clim.reset_index()
    my_clim["P_F_c"] = fill_na(np.array(my_clim.P_F))
    my_clim["LE_all_c"] = fill_na(np.array(my_clim.LE_F_MDS))
    dfm = pd.merge(df,my_clim[["doy","P_F_c","LE_all_c"]],on="doy",how="left")
        
    
    p_in = fill_na2(np.array(df.P_F),np.array(dfm.P_F_c))
    et_out = fill_na2(et_summer * 18/1000 * 60*60*24,np.array(dfm["LE_all_c"] / 44200 * 18/1000 * 60*60*24))
    doy_summer = np.array(df["doy"])
    #%%
    if np.mean(et_out) > np.mean(p_in):
        print("Lateral flow needed")
        site_result[site_id] = "Long-term ET exceeds P"

        continue
    #%%
    yeardf = df.groupby("year").sum(numeric_only=True).reset_index()
    yearET = yeardf.LE_F_MDS / 44200 * (18/1000) * (60*60*24)
    bad_year = yeardf.loc[yeardf.P_F < yearET].year
    #%%
    to_replace = df.year.isin(bad_year)
    p_in[to_replace] = dfm.P_F_c[to_replace]
    et_out[to_replace] = dfm.LE_all_c[to_replace] / 44200 * 18/1000 * 60*60*24
    
    #%%
    # opposite_peak = topday - 180
    # if opposite_peak < 0:
    #     opposite_peak += 365
    #%%
    
    wbi = 0
    waterbal_raw = np.zeros(len(doy_summer))
    for dayi in range(len(p_in)):
        waterbal_raw[dayi] = wbi
        wbi += p_in[dayi] - et_out[dayi] #- 0.01*wbi
        wbi = min(0,wbi)
        #if dayi == opposite_peak:
        #    wbi = 0
    waterbal_corr = 1*waterbal_raw
    waterbal_corr[to_replace] = np.nan
    # latflow_daily = np.clip(waterbal_raw[:-365] - waterbal_raw[365:],0,np.inf)/365
    # latflow_daily_2 = 0*latflow_daily
    # latflow_daily_2[182:] = latflow_daily[:-182]
    # latflow_daily_2[:182] = np.mean(latflow_daily[:182])
    #%%
    # if np.mean(latflow_daily_2 > 0) > 0.25:
    #     print("Lateral flow needed")
    #     continue    
    
    # wbi = 0#-300
    # waterbal_corr = np.zeros(len(doy_summer))
    # for dayi in range(len(p_in)):
    #     waterbal_corr[dayi] = wbi
    #     wbi += p_in[dayi] - et_out[dayi] + latflow_daily_2[min(dayi,len(latflow_daily)-1)]
    #     wbi = min(0,wbi)
    #%%
   
    #%%
    waterbal_corr[np.isnan(et_summer)] = np.nan
    
    
    smc_summer = np.array(meancols(df,'SWC'))
    
    sinterp = np.interp(smc_summer,np.sort(smc_summer),np.sort(waterbal_corr))
    
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
    gpp_summer = np.array(df["GPP_NT_VUT_REF"])
    #airt_summer[airt_summer < 0] = np.nan
    gpp_summer[gpp_summer < 0] = np.nan
    
    nee_qc = np.array(df.NEE_VUT_REF_QC)
    gpp_summer[nee_qc < 0.5] = np.nan
    
    is_summer = (doy_summer >= summer_start)*(doy_summer <= summer_end)
    is_late_summer = (doy_summer >= topday)*(doy_summer <= summer_end)
    
    pet = petVnum/(sV+gammaV)

    #%%
    if np.sum(np.isfinite(gpp_summer)) < (25):
        print("Not enough GPP")
        site_result[site_id] = "Not enough data"

        continue
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
    rain_dict[site_id] = [rain_fake[is_late_summer],np.array(df.year)[is_late_summer]]
    #not sure whether using rain over entire or late summer is more appropriate
#    rain_dict[site_id] = [rain_fake[is_summer],np.array(df.year)[is_summer]]

    
    #is_summer = is_late_summer
    #summer_start = topday
    
    
    wbs = waterbal_corr[is_summer]
    wbs[doy_summer[is_summer]==summer_start] = 0
    alldd = []
    ddi = []

    for j in range(1,len(waterbal_corr)):
        if waterbal_corr[j] < max(waterbal_corr[max(j-6,0):j]): 
            ddi.append(j)
        else:
            alldd.append(ddi)
            ddi = []
    #%%
    ddlen = np.array([len(x) for x in alldd])
    #dfi["ddmax"] = np.max(ddlen)
    #dfi["ddmean"] = np.mean(ddlen[ddlen >= 1])
    #dfi["dd95"] = np.quantile(ddlen[ddlen >= 1],0.95)
    dd_dict1[site_id] = ddlen
  
    #%%
    slen = 5
    
    #%%
    smc_smooth = 0*waterbal_corr#[:-5]
    for i in range(5):
        smc_smooth[:-5] += waterbal_corr[i:len(smc_summer)-5+i]/5
    smc_smooth[-5:] = waterbal_corr[-5:]
        #%%
    wbs = smc_smooth[is_summer]
    wbs[doy_summer[is_summer]==summer_start] = 0
    alldd = []
    ddi = []

    for j in range(1,len(wbs)):
        if wbs[j] <= wbs[j-1]: 
            ddi.append(j)
        else:
            alldd.append(ddi)
            ddi = []
    ddlen = np.array([len(x) for x in alldd])

    dd_dict_wbal_smooth[site_id] = ddlen
    #%%
    
    #%%
    df_to_fit = pd.DataFrame({"date":df.date,"airt":airt_summer,"year":df.year,
                              "par":par_summer,"cosz":cosz,
                              "cond":daily_cond,"gpp":gpp_summer,
                              "doy":doy_summer,"vpd":vpd_summer,
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
                              #"sinterp":sinterp,
                              "gpp_unc":-df.NEE_VUT_REF_RANDUNC/df.NEE_VUT_REF,
                              #"PET":pet
                              })
                              #"LE_unc":df.LE_RANDUNC/44200})
    df_to_fit = df_to_fit.loc[is_summer].dropna()
    
    df_to_fit = df_to_fit.loc[df_to_fit.rain==0]
    df_to_fit = df_to_fit.loc[df_to_fit.rain_prev==0]

#    df_to_fit = df_to_fit.loc[(df_to_fit.doy >= topday)*(df_to_fit.vpd >= 0.5)].copy()
#%%
    df_to_fit = df_to_fit.loc[df_to_fit.doy >= topday].copy()
    df_to_fit = df_to_fit.loc[(df_to_fit.et_unc / df_to_fit.ET) <= 0.2].copy()
    
    
    #%%
    gpp_qc = np.array(df_to_fit.gpp)
    gpp_qc[(df_to_fit.gpp_unc <= 0) | (df_to_fit.gpp_unc > 0.2)] = np.nan
    #df_to_fit = df_to_fit.loc[(df_to_fit.gpp_unc >= 0)*(df_to_fit.gpp_unc < 0.15)].copy()
    df_to_fit["gpp_qc"] = gpp_qc
    
    #df_to_fit = df_to_fit.loc[df_to_fit.rain_prev==0]
    #df_to_fit = df_to_fit.loc[df_to_fit.vpd >= 0.5]
#%%
    year_count = df_to_fit.groupby("year").count()
    year_count["ETcount"] = year_count["ET"]*1
    df_to_fit = pd.merge(df_to_fit,year_count["ETcount"],on="year",how='left')
    df_to_fit = df_to_fit.loc[df_to_fit["ETcount"] >= 10].copy()

    #%%
    if len(df_to_fit) < 25:
        print("Not enough data")
        site_result[site_id] = "Not enough data in growing season"

        continue
    #%%
    if sum(np.isfinite(gpp_qc)) < 25:
        print("Not enough data")

        site_result[site_id] = "Not enough data in growing season"
        continue
    #%%
    g_samp = np.array(df_to_fit.cond)
    gpp_samp = np.array(df_to_fit.gpp_qc)
    par_samp = np.array(df_to_fit.par)

    obs_gppmax = np.max(gpp_samp)
    
    doy_samp = np.array(df_to_fit.doy)
    #rising_lai = np.clip(doy_samp-topday,-np.inf,0)
    #falling_lai = -np.clip(doy_samp-topday,0,np.inf)
    
    rising_lai = np.clip(doy_samp,0,topday)
    falling_lai = np.clip(doy_samp,topday,np.inf)
    #%%
    df_to_fit["LAI_spring"] = rising_lai
    df_to_fit["LAI_fall"] = falling_lai

    #%%
    # gpp_g_cor = scipy.stats.spearmanr(g_samp[np.isfinite(gpp_samp)],gpp_samp[np.isfinite(gpp_samp)])
    # if gpp_g_cor.pvalue > 0.05 or gpp_g_cor.correlation < 0:
    #     print("GPP not limited")
    #     continue
    #%%
    df_hi_cond = []
    for yearJ in pd.unique(df_to_fit.year):
        dataJ = df_to_fit.loc[df_to_fit.year == yearJ]
        year_cutoff = np.quantile(dataJ.cond,0.67)
        year_high = dataJ.loc[dataJ.cond > year_cutoff]
        df_hi_cond.append(year_high)
    df_hi_cond = pd.concat(df_hi_cond).reset_index()
    
    #%%
    #df_hi_cond = df_to_fit.loc[df_to_fit.cond > np.quantile(df_to_fit.cond,0.67)].copy()
    #hi_mod = smf.ols("np.log(gpp_qc) ~ np.log(par) + np.log(LAI_spring) + np.log(LAI_fall)",data=df_hi_cond,missing="drop").fit()
#    hi_mod = smf.ols("np.log(gpp_qc) ~ par + airt + np.power(airt-25,2)",data=df_hi_cond,missing="drop").fit()
    hi_mod = smf.ols("np.log(gpp_qc) ~ np.log(par) + C(year)",data=df_hi_cond,missing="drop").fit()

    #%%
    gppmax = np.exp(hi_mod.predict(df_to_fit))

   #%%
   # if hi_mod.pvalues[1] < 0.05 and hi_mod.params[1] > 0:
    #gpp_mean = np.mean(df_to_fit.loc[df_to_fit.cond > 0.2].gpp)
    # def gpp_opt(pars):
    #     #b_par, b_airt, b_airt2 = coefs[:3]
    #     #year_coef = coefs[3:]
    #     gppmax = np.exp(pars[1] + np.log(par_samp)*pars[2] + pars[3]*rising_lai + pars[4]*falling_lai)
    #     gpp_pred = gppmax*(1-np.exp(-g_samp*pars[0]/gppmax))
    #     return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    # fit0 = np.zeros(5)
    # fit0[0] = 50
    # fit0[1:] = hi_mod.params
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    
    # pars = gpp_optres.x
    # gppmax = np.exp(pars[1] + np.log(par_samp)*pars[2]  + pars[3]*rising_lai + pars[4]*falling_lai)
    
    #%%
    
    def gpp_opt(pars):
        #b_par, b_airt, b_airt2 = coefs[:3]
        #year_coef = coefs[3:]
        gpp_pred = gppmax*(1-np.exp(-g_samp*pars[0]/gppmax))
        return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    fit0 = np.zeros(1)
    fit0[0] = 50
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    #%%
    
    #gppmax = np.exp(pars[1] + np.log(par_samp)*pars[2]  + pars[3]*rising_lai + pars[4]*falling_lai)
        
    # else:
    #     def gpp_opt(pars):
    #         #b_par, b_airt, b_airt2 = coefs[:3]
    #         #year_coef = coefs[3:]
    #         gppmax = np.exp(pars[1] + pars[2]*rising_lai + pars[4]*falling_lai)
    #         gpp_pred = gppmax*(1-np.exp(-g_samp*pars[0]/gppmax))
    #         return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
        
    #     fit0 = np.zeros(2)
    #     fit0[0] = 50
    #     fit0[1] = np.nanmax(gpp_samp)
    #     gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm",x_scale=np.abs(fit0))
        
        
    #     pars = gpp_optres.x
    #     gppmax = pars[1]
        
      #%%  
        
        
    
    df_to_fit["gppmax"] = gppmax

    kg = df_to_fit["gppmax"]/gpp_optres.x[0]
    #df_to_fit["res_cond"] = gpp_optres.x[1]
    
    
    # plt.figure()
    # plt.plot(dfi.cond/kg, dfi.gpp/dfi["gppmax"],'o'); 
    # plt.plot(xcond,1-np.exp(-xcond))
    # plt.xlim(0,5)
    # plt.ylim(0,2)
    # #%%
    # plt.figure()
    # plt.plot(dfi.cond, dfi.gpp,'o'); 
    # plt.xlim(0,0.5)
    #%%
    gpp_pred = gppmax*(1-np.exp(-g_samp/kg))
    gpp_r2 = 1-np.nanmean((gpp_pred-gpp_samp)**2)/np.nanvar(gpp_samp)
    df_to_fit["gppR2"] = gpp_r2
    #gpp_r2_null = 1-np.mean((df_to_fit.gppmax-df_to_fit.gpp)**2)/np.var(df_to_fit.gpp)
#    gpp_r2_null = np.corrcoef(gpp_samp,df_to_fit.gppmax)[0,1]**2

    #%%
    # if (1-gpp_r2)/(1-gpp_r2_null) > 0.9:
    #     print("No GPP limitation")
    #     continue
    #%%
    #plt.figure()
    #plt.plot(df_to_fit.doy,df_to_fit.gpp-gpp_pred,'.')
    #%%
    df_to_fit["res_gpp"] = df_to_fit.gpp-gpp_pred
    #%%
    df_fit_doy = df_to_fit.groupby("doy").mean(numeric_only=True).rename(columns={"res_gpp":"doy_res"})
    #%%
    df_to_fit = pd.merge(df_to_fit,df_fit_doy["doy_res"],on="doy",how="left")

    #%%
    #dfi = df_to_fit.loc[df_to_fit.waterbal < np.nanmedian(df_to_fit.waterbal)].copy()

    dfi = df_to_fit.loc[(df_to_fit.doy >= topday)].copy()
   # dfi = df_to_fit.copy()

    #%%
    wbal_samp = np.array(dfi["waterbal"])/1000
    vpd_samp = np.array(dfi.vpd)
    
    et_samp = np.array(dfi['ET'])
    k_samp = np.array(dfi["gppmax"]/gpp_optres.x[0])
    
    
    gasvol_fac_samp = np.array(dfi['gasvol_fac'])
    petVnum_samp = np.array(dfi['petVnum'])
    myga_samp = np.array(dfi['myga'])
    sV_samp = np.array(dfi['sV'])
    g_samp = np.array(dfi["cond"])
    
    #year_min = np.array(df_omit_nan.groupby("year").min()["waterbal"])
    
    fac1 = myga_samp/(22.4*gasvol_fac_samp/1000)
    g_adj = g_samp/np.sqrt(2*zsoil_mol*k_samp/(vpd_samp/100))

   # yearmat = np.array(pd.get_dummies(dfi.year))
    #%%
   
    
    #%%
    
    def tofit(pars):
        tau = pars[0]
        slope = np.sqrt(1/(tau*(60*60*24)))
    
        smin = pars[1]
        s_adj = np.clip(wbal_samp- smin,0,width)
        final_cond = np.clip(slope*np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100)),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
    
    fit0 = np.zeros(2)
    fit0[0] = 30
    fit0[1] = np.quantile(wbal_samp,0.05) #np.array(dfi.groupby("year").min(numeric_only=1).waterbal/1000)
    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    tau = pars[0]
    slope = np.sqrt(1/(tau*(60*60*24)))
    
    smin = pars[1]
    s_adj = np.clip(wbal_samp- smin,0,width)
    final_cond = np.clip(slope*np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100)),0,gmax)
    et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
          
      
    # dfi["tau"] = tau
    # dfi["smin"] = smin
    # dfi["gmax"] = gmax
    # dfi["etr2_tau"] = 1-np.mean((et_out-et_samp)**2)/np.var(et_samp)
    #%%
    # plt.figure()
    # plt.plot(wbal_samp-smin,g_adj,'.')
    # xarr = np.linspace(0,1,500)
    # plt.plot(xarr,np.sqrt(np.clip(xarr,0,width)/tau/(60*60*24)),'k',linewidth=3)
    # plt.title(site_id+", tau = "+str(np.round(tau,1))+" days")
    #%%
    # yi = dfi.year==2007
    # plt.plot(wbal_samp[yi]+0.2,g_adj[yi],'o')
    # xarr = np.linspace(0,0.15,500)
    # plt.plot(xarr,np.sqrt(np.clip(xarr,0,width)/12/(60*60*24)),'k',linewidth=3)
    # ##yi = dfi.year==2008
    # #plt.plot(wbal_samp[yi]+0.2,g_adj[yi],'o')
    # yi = dfi.year==2012
    # plt.plot(wbal_samp[yi]+0.16,g_adj[yi],'o')
    #%%
    dfi["tau"] = tau
    #dfi["width"] = width
    dfi["smin"] = smin
    dfi["gpp_slope"] = gpp_optres.x[0]
    #dfi["gpp_slope"] = gpp_optres.x[1]/gpp_optres.x[0]
    dfi["g_adj"] = g_adj
    dfi["et_tau"] = et_out
    dfi["SITE_ID"] = site_id
    dfi["etr2_smc"] = 1-np.mean((et_out-et_samp)[:]**2)/np.var(et_samp)
    #%%
    def tofit(pars):
        slope0 = pars[0]
        #gmax = pars[1]
        final_cond = np.clip(slope0*np.sqrt(k_samp*zsoil_mol/(vpd_samp/100)),0,gmax)
        et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        return (et_out-et_samp)
    
    fit0 = np.zeros(1)
    fit0[0] = np.median(g_samp / np.sqrt(k_samp*zsoil_mol/(vpd_samp/100)))
    #%%
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    slope0 = pars[0]
    final_cond = np.clip(slope0*np.sqrt(k_samp*zsoil_mol/(vpd_samp/100)),0,gmax)
    et_null = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    dfi["etr2_null"] = 1-np.mean((et_null-et_samp)[:]**2)/np.var(et_samp)
    dfi["et_null"] = et_null
#%%
    #dfi = dfi.sort_values("waterbal")
    #m5 = len(dfi) / 5
    #xmed = np.array(dfi.waterbal)
    # x1 = np.linspace(np.min(wbal_samp), np.max(wbal_samp),21)
    # y1 = np.zeros(20)
    # for zi in range(20):
    #     seli = (wbal_samp >= x1[zi])*(wbal_samp < x1[zi+1])
    #     y1[zi] = np.median(g_adj[seli])
    #%%\

    #%%
    # ddlen = np.array([len(x) for x in alldd])
    # dfi["ddmax_wbal_smooth"] = np.max(ddlen)
    # dfi["ddmean_wbal_smooth"] = np.mean(ddlen[ddlen >= 1])
    # dfi["dd95_wbal_smooth"] = np.quantile(ddlen[ddlen >= 1],0.95)

    dfi["mat_data"] = np.nanmean(airt_summer)
    dfi["map_data"] = np.nanmean(p_in)
    #dfi["mgsp_data"] = np.nanmean(p_in[late_summer])
    dfi["mean_netrad"] = np.nanmean(myrn)
    #dfi["mean_netrad"] = np.nanmean(myrn[late_summer])

    dfi["gs_netrad"] = np.nanmean(myrn[is_late_summer])
    dfi["pet_year"] = np.nanmean(pet)
    dfi["pet_late_summer"] = np.nanmean(pet[is_late_summer])

    #%%
    dfi["snow_frac"] = np.sum(p_in*(airt_summer < 0))/np.sum(p_in)
    
    new_waterbal = 1*waterbal_corr
    new_waterbal[doy_summer == topday] = 0
    
    wb_dict[site_id] = new_waterbal[is_late_summer]

    dfi["gs_peak"] = topday
    dfi["gs_end"] = summer_end
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
cors = []
for x in df1.SITE_ID:
    subI = all_results.loc[all_results.SITE_ID==x]
    x1 = np.array(subI.waterbal)
    y1 = np.array(subI.g_adj)
    goodi = np.isfinite(x1+y1)
    corx = scipy.stats.spearmanr(x1[goodi],y1[goodi])
    cors.append(corx)
#%%
#df1 = df1.loc[np.array([x.correlation for x in cors]) > 0.33]
#%%
def qt_gt1(x,q):
    return np.quantile(x[x >= 1],q)
def mean_gt1(x):
    return np.mean(x[x >= 1])
#%%
#df1["ddmax_wbal_smooth"] = [qt_gt1(dd_dict_wbal_smooth[x],0.95) for x in df1.SITE_ID]
df1["ddmax_wbal_smooth"] = [np.max(dd_dict_wbal_smooth[x]) for x in df1.SITE_ID]
#df1["ddmax_wbal_smooth"] = [mean_gt1(dd_dict_wbal_smooth[x]) for x in df1.SITE_ID]

#%%
#df1 = df1.loc[(1-df1.etr2_smc)/(1-df1.etr2_null) < 0.95]
#df1 = df1.loc[df1.etr2_smc - df1.etr2_null > 0.05]
#df1 = df1.loc[df1.snow_frac < 0.5]
#df1 = df1.loc[df1.etr2_smc-df1.etr2_null > 0.05]

#df1 = pd.merge(df1,bif_forest,on="SITE_ID",how="left")
#df1 = df1.loc[df1.mat_data > 3]
#df1 = df1.loc[df1.gppR2 > 0]

#df1 = df1.loc[df1.site_count >= 100]
#df1 = df1.loc[df1.year_count >= 3]

#df1["dd90"] = [np.mean(dd_dict_wbal_smooth[x][dd_dict_wbal_smooth[x] > 10]) for x in df1.SITE_ID]
#%%
#reg0 = sm.OLS(df1.tau,df1.ddmax_wbal_smooth).fit()
#reg1 = sm.OLS(df1.tau,sm.add_constant(df1.ddmax_wbal_smooth)).fit()

df_meta = df1.copy()
#df_meta = pd.merge(df1,metadata,how="left",left_on="SITE_ID",right_on="fluxnetid")
#dfmean = all_results.groupby("SITE_ID").mean(numeric_only=1)

#%%
# biome_dict = {"SAV":"Savanna",
#               "EBF":"Evergreen broadleaf forest",
#               "ENF":"Evergreen needleleaf forest",
#               "GRA":"Grassland",
#               "DBF":"Deciduous broadleaf forest",
#               "CSH":"Closed shrubland",
#               "MF":"Mixed forest",
#               "WSA":"Woody savanna"}
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
plot_soil_dd = 0

if plot_soil_dd:
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    
    line1, = ax.plot([0,180],[0,180],"k",label="1:1 line")
    line2, = ax.plot([0,180],np.array([0,180])*reg1.params[1]+reg1.params[0],"b--",label="Regression line\n($R^2$ = 0.45)")
    #plt.plot([0,150],np.array([0,150])*reg0.params[0],"b--",label="Regression line\n($R^2$ = 0.39)")
    leg1 = ax.legend(loc="upper left")
    
    points_handles = []
    for i in range(len(biome_list)):
        subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
        pointI, = ax.plot(subI.ddmax_wbal_smooth,subI.tau,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
    
    ax.set_xlabel("Maximum drydown length (days)",fontsize=24)
    ax.set_ylabel(r"$\tau$ (days)",fontsize=24)
    
    fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.46,0),ncols=2 )
    ax.add_artist(leg1)
    
    plt.xlim(0,160)
    plt.ylim(0,160)

#%%
plot_gadj = 0

if plot_gadj:
    yscaler = np.sqrt(zsoil_mol)
    molm2_to_mm = 18/1000
    s2day = 60*60*24
    
    site_id = "US-MMS"
    df2 = all_results.loc[all_results.SITE_ID==site_id]
    
    plt.figure()
    plt.plot((df2.waterbal/1000)*zsoil_mol*molm2_to_mm,(df2.g_adj*yscaler*np.sqrt(molm2_to_mm))**2*s2day,'o',alpha=1,label=r"US-Me5, $\tau$ = 74 days")
    xarr = np.linspace(0,1000,500)
    #tau_overall = np.median(df2.tau)
    tauI = df2.tau.iloc[0]
    print(tauI)
    #width=df2.width.iloc[0]
    plt.plot(xarr+df2.smin.iloc[0]*zsoil_mol*molm2_to_mm,(np.clip(xarr,0,width)/tauI/(60*60*24))*s2day,'k',linewidth=2)
    
    # tau2 = 40.6
    # smin2 = -0.5
    # plt.plot(xarr+smin2*zsoil_mol*molm2_to_mm,(np.clip(xarr,0,width)/tauI/(60*60*24)),'g',linewidth=3)
    
    
    site_id = "DE-Hai"
    
    df2 = all_results.loc[all_results.SITE_ID==site_id]
    
    plt.plot((df2.waterbal/1000)*zsoil_mol*molm2_to_mm,(df2.g_adj*yscaler*np.sqrt(molm2_to_mm))**2*s2day,'o',alpha=1,label=r"US-LWW, $\tau$ = 20 days")
    xarr = np.linspace(0,1000,500)
    
    #tau_overall = np.median(df2.tau)
    tauI = df2.tau.iloc[0]
    print(tauI)
    #width=df2.width.iloc[0]
    plt.plot(xarr+df2.smin.iloc[0]*zsoil_mol*molm2_to_mm,(np.clip(xarr,0,width)/tauI/(60*60*24))*s2day,'k',linewidth=2)
    plt.xlabel("$Z_{soil}*s$ (mm) (0 = saturation)",fontsize=30)
    #plt.ylabel("Adjusted conductance $(s^{0.5})$")
    plt.ylabel("$g_{adj}$ (mm/day)",fontsize=30)
    
    plt.legend(fontsize=24,framealpha=1)
    
    plt.xlim(-220,0)
    plt.ylim(0,20)
    #plt.title(site_id+", tau 
#%%

yscaler = np.sqrt(zsoil_mol)
molm2_to_mm = 18/1000
s2day = 60*60*24

site_id = "US-Me5"
df2 = all_results.loc[all_results.SITE_ID==site_id]

plt.figure()
plt.plot((df2.waterbal/1000)*zsoil_mol*molm2_to_mm,df2.cond,'o',alpha=1,label=r"US-Me5, $\tau$ = 74 days")
xarr = np.linspace(0,1000,500)
tauI = df2.tau.iloc[0]
print(tauI)

site_id = "US-LWW"

df2 = all_results.loc[all_results.SITE_ID==site_id]

plt.plot((df2.waterbal/1000)*zsoil_mol*molm2_to_mm,df2.cond,'o',alpha=1,label=r"US-LWW, $\tau$ = 20 days")
xarr = np.linspace(0,1000,500)

tauI = df2.tau.iloc[0]
print(tauI)
plt.xlabel("$Z_{soil}*s$ (mm) (0 = saturation)",fontsize=30)
plt.ylabel("g $(mol/m^2/s)$",fontsize=30)

plt.legend(fontsize=24,framealpha=1)

plt.ylim(0,0.4)
#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/sample_sites2.svg",bbox_inches="tight")

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

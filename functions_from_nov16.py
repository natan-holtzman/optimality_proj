# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:27:50 2023

@author: natan
"""


import numpy as np
#import pymc as pm
import pandas as pd
import scipy.optimize
import statsmodels.formula.api as smf

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


def prepare_df(fname, site_id, bif_forest):
    df = pd.read_csv(fname,parse_dates=["TIMESTAMP"])
    df[df==-9999] = np.nan
    
    df["date"] = df["TIMESTAMP"].dt.date
    df["hour"] = df["TIMESTAMP"].dt.hour
    df["doy"] = df["TIMESTAMP"].dt.dayofyear
    site_id = fname.split("\\")[-1].split('_')[1]
    #print(site_id)
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
        return "Not enough data"
        
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
        return "Long-term ET exceeds P"

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
    df_to_fit = df_to_fit_full.loc[is_summer].dropna()
    
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
    
    df_to_fit["mat_data"] = np.nanmean(airt_summer)
    df_to_fit["map_data"] = np.nanmean(p_in)
    #dfi["mgsp_data"] = np.nanmean(p_in[late_summer])
    df_to_fit["mean_netrad"] = np.nanmean(myrn)
    df_to_fit["SITE_ID"] = site_id


    return df_to_fit, rain_for_dict, np.nanmean(airt_summer), np.nanmean(rain_summer), df_to_fit_full
#%%
def fit_gpp(df0):
    df_to_fit = df0.copy()
    g_samp = np.array(df_to_fit.cond)
    gpp_samp = np.array(df_to_fit.gpp_qc)
    par_samp = np.array(df_to_fit.par)

    obs_gppmax = np.max(gpp_samp)
    
    doy_samp = np.array(df_to_fit.doy)
    
    #%%
    df_hi_cond = []
    for yearJ in pd.unique(df_to_fit.year):
        dataJ = df_to_fit.loc[df_to_fit.year == yearJ]
        year_cutoff = np.quantile(dataJ.cond,0.67)
        year_high = dataJ.loc[dataJ.cond > year_cutoff]
        df_hi_cond.append(year_high)
    df_hi_cond = pd.concat(df_hi_cond).reset_index()
    
    #%%
    hi_mod = smf.ols("np.log(gpp_qc) ~ np.log(par) + C(year)",data=df_hi_cond,missing="drop").fit()

    #%%
    gppmax = np.exp(hi_mod.predict(df_to_fit))

    #%%
    
    def gpp_opt(pars):
        #b_par, b_airt, b_airt2 = coefs[:3]
        #year_coef = coefs[3:]
        gpp_pred = gppmax*(1-np.exp(-g_samp*pars[0]/gppmax))
        return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    fit0 = np.zeros(1)
    fit0[0] = 50
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    
    df_to_fit["gppmax"] = gppmax

    kg = df_to_fit["gppmax"]/gpp_optres.x[0]
    df_to_fit["kgpp"] =kg
    df_to_fit["gpp_slope"] = gpp_optres.x[0]

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
    # df_to_fit["res_gpp"] = df_to_fit.gpp-gpp_pred
    # #%%
    # df_fit_doy = df_to_fit.groupby("doy").mean(numeric_only=True).rename(columns={"res_gpp":"doy_res"})
    # #%%
    # df_to_fit = pd.merge(df_to_fit,df_fit_doy["doy_res"],on="doy",how="left")
    return df_to_fit
#%%


def fit_tau(dfi):


    
    wbal_samp = np.array(dfi["waterbal"])/1000
    vpd_samp = np.array(dfi.vpd)
    
    et_samp = np.array(dfi['ET'])
    k_samp = np.array(dfi["kgpp"]) #np.array(dfi["gppmax"]/gpp_optres.x[0])
    
    
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
    
    #%%
    dfi["tau"] = tau
    #dfi["width"] = width
    dfi["smin"] = smin
    #dfi["gpp_slope"] = gpp_optres.x[0]
    #dfi["gpp_slope"] = gpp_optres.x[1]/gpp_optres.x[0]
    dfi["g_adj"] = g_adj
    dfi["et_tau"] = et_out
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
    return dfi
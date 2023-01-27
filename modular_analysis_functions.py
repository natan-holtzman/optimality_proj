# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:14:25 2023

@author: natan
"""

import numpy as np
#import pymc as pm
import pandas as pd
import scipy.optimize
import statsmodels.formula.api as smf

#%%

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
def fit_tau(df_to_fit):

    dfi = df_to_fit.copy()

    #%%
    wbal_samp = np.array(dfi["waterbal"])/1000
    vpd_samp = np.array(dfi.vpd)
    
    et_samp = np.array(dfi['ET'])
    k_samp = np.clip(np.array(dfi["gppmax"]/dfi["gpp_slope"]),0,np.inf)
    
    
    gasvol_fac_samp = np.array(dfi['gasvol_fac'])
    petVnum_samp = np.array(dfi['petVnum'])
    myga_samp = np.array(dfi['myga'])
    sV_samp = np.array(dfi['sV'])
    g_samp = np.array(dfi["cond"])
        
    fac1 = myga_samp/(22.4*gasvol_fac_samp/1000)
    g_adj = g_samp/np.sqrt(2*zsoil_mol*k_samp/(vpd_samp/100))

   # gres = dfi["res_cond"].iloc[0]
   # yearmat = np.array(pd.get_dummies(dfi.year))
    #%%
    
    def tofit(pars):
        tau = pars[0]
        #tau_s = tau*(60*60*24)
        slope = np.sqrt(1/(tau*(60*60*24)))
    
        smin = pars[1]
        s_adj = np.clip(wbal_samp- smin,0,width)
        final_cond = np.clip(slope*np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100)),0,gmax)
        #final_cond = np.sqrt(2*k_samp*zsoil_mol*s_adj/(vpd_samp/100)/tau_s + gres**2) #- gres
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
    
    # plt.figure()
    # plt.plot(wbal_samp-smin,g_adj,'.')
    # xarr = np.linspace(0,1,500)
    # plt.plot(xarr,np.sqrt(np.clip(xarr,0,width)/tau/(60*60*24)),'k',linewidth=3)
    # plt.title(site_id+", tau = "+str(np.round(tau,1))+" days")
    
    dfi["tau"] = tau
    #dfi["width"] = width
    dfi["smin"] = smin
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
    
    cond_optres = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
    pars = cond_optres.x
    slope0 = pars[0]
    final_cond = np.clip(slope0*np.sqrt(k_samp*zsoil_mol/(vpd_samp/100)),0,gmax)
    et_null = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
    dfi["etr2_null"] = 1-np.mean((et_null-et_samp)[:]**2)/np.var(et_samp)
    dfi["et_null"] = et_null

    return dfi
#%%
def fit_gpp(df0):
    df_to_fit = df0.copy()
    g_samp = np.array(df_to_fit.cond)
    gpp_samp = np.array(df_to_fit.gpp_qc)
    #par_samp = np.array(df_to_fit.par)
    #obs_gppmax = np.max(gpp_samp)
    
    doy_samp = np.array(df_to_fit.doy)
    #%%
    # rising_lai = np.clip(doy_samp,0,topday)
    # falling_lai = np.clip(doy_samp,topday,np.inf)
    
    # df_to_fit["LAI_spring"] = rising_lai
    # df_to_fit["LAI_fall"] = falling_lai

    #%%
    # gpp_g_cor = scipy.stats.spearmanr(g_samp[np.isfinite(gpp_samp)],gpp_samp[np.isfinite(gpp_samp)])
    # if gpp_g_cor.pvalue > 0.05 or gpp_g_cor.correlation < 0:
    #     print("GPP not limited")
    #     continue
    #%%
    qpast = 0.75
    df_hi_cond = []
    for yearJ in pd.unique(df_to_fit.year):
        dataJ = df_to_fit.loc[df_to_fit.year == yearJ]
        year_cutoff = np.quantile(dataJ.cond,qpast)
        year_high = dataJ.loc[dataJ.cond > year_cutoff]
        df_hi_cond.append(year_high)
    df_hi_cond = pd.concat(df_hi_cond).reset_index()
    par_slope_guess = np.mean(df_hi_cond.gpp) / np.mean(df_hi_cond.par)
    #df_hi_cond = df_to_fit[df_to_fit.cond > np.quantile(df_to_fit.cond,qpast)].copy()
    #%%

    #%%
 #   hi_mod = smf.ols("np.log(gpp_qc) ~ doy + np.log(np.clip(par,100,2000)) + np.log(airt + 273) + C(year)",data=df_hi_cond,missing="drop").fit()
 #   hi_mod = smf.ols("np.log(gpp_qc) ~ np.log(np.clip(par,100,2000)) + C(year)",data=df_hi_cond,missing="drop").fit()

#    hi_mod = smf.ols("gpp_qc ~ doy + par + airt",data=df_hi_cond,missing="drop").fit()
#    hi_mod = smf.ols("np.log(gpp_qc) ~ np.log(par) + C(year) + np.log(LAI_spring+180) + np.log(LAI_fall+180)",data=df_hi_cond,missing="drop").fit()
#%%
    #df_to_fit = df_to_fit.loc[df_to_fit.year.isin(df_hi_cond.year)].copy()
    g_samp = np.array(df_to_fit.cond)
    gpp_samp = np.array(df_to_fit.gpp_qc)
   
    #obs_gppmax = np.max(gpp_samp)
    doy_samp = np.array(df_to_fit.doy)
    
#    gppmax = np.exp(hi_mod.predict(df_to_fit))

    #%%
    # def gpp_opt(pars):
    #     #b_par, b_airt, b_airt2 = coefs[:3]
    #     #year_coef = coefs[3:]
    #     gpp_pred = gppmax*(1-np.exp(-g_samp*pars[0]/gppmax))
    #     return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    # fit0 = np.zeros(1)
    # fit0[0] = 150
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
       
    # df_to_fit["gppmax"] = gppmax
    # kg = df_to_fit["gppmax"]/gpp_optres.x[0]
    #%%
    #gpp_stack = np.stack((np.log(np.clip(dfgpp.par,100,2000)), np.log(dfgpp.airt + 273)),1)
    yearmat = np.array(pd.get_dummies(df_to_fit.year))
    #%%
    par_samp = np.array(df_to_fit.par)
    def gpp_opt(pars):
        #b_par, b_airt, b_airt2 = coefs[:3]
        #year_coef = coefs[3:]
        #gppmax_pred = np.clip(df_to_fit.par,100,2000)**pars[1] *  (df_to_fit.airt + 273)**pars[2]  *  np.exp(yearmat.dot(pars[3:]))
#        year_coef = yearmat.dot(pars)
        year_coef = pars[0]
        res_cond = max(0,pars[1])

        gppmax_pred = year_coef*par_samp
        gpp_pred = gppmax_pred*(1-np.exp(-(g_samp-res_cond)*100/gppmax_pred))
        return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    fit0 = np.zeros(2)#yearmat.shape[1])
    fit0[0] = par_slope_guess
    fit0[1] = np.min(g_samp)
    #fit0[2] = 100
    gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    #gppmax *= gpp_optres.x[1]
    pars = gpp_optres.x
    year_coef =  pars[0]#yearmat.dot(pars[:])
    res_cond = max(0,pars[1])

    gppmax = year_coef*par_samp

    df_to_fit["gppmax"] = gppmax
    kg = df_to_fit["gppmax"]/100#gpp_optres.x[2]
    df_to_fit["res_cond"] = res_cond
 #%%   
    # def gpp_opt(pars):
    #     #b_par, b_airt, b_airt2 = coefs[:3]
    #     #year_coef = coefs[3:]
    #     gppmax_pred = np.clip(df_to_fit.par,100,2000)**pars[1] *  (df_to_fit.airt + 273)**pars[2]  * np.exp(pars[3])#*  np.exp(yearmat.dot(pars[3:]))
    #     gpp_pred = gppmax_pred*(1-np.exp(-g_samp*pars[0]/gppmax_pred))
    #     return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    # fit0 = np.zeros(4)
    # fit0[0] = 100
    # fit0[1] = hi_mod.params[1]
    # fit0[2] = hi_mod.params[2]
    # fit0[3] = hi_mod.params[0]
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    # #gppmax *= gpp_optres.x[1]
    # pars = gpp_optres.x
    # gppmax = np.clip(df_to_fit.par,100,2000)**pars[1] *  (df_to_fit.airt + 273)**pars[2]  *  np.exp(pars[3])# np.exp(yearmat.dot(pars[3:]))

    # df_to_fit["gppmax"] = gppmax
    # kg = df_to_fit["gppmax"]/gpp_optres.x[0]
    #%%
    # def gpp_opt(pars):
    #     #b_par, b_airt, b_airt2 = coefs[:3]
    #     #year_coef = coefs[3:]
    #     gppmax_pred = np.clip(df_to_fit.par,100,2000)**pars[1]  * np.exp(pars[2])#*  np.exp(yearmat.dot(pars[3:]))
    #     gpp_pred = gppmax_pred*(1-np.exp(-g_samp*pars[0]/gppmax_pred))
    #     return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    # fit0 = np.zeros(3)
    # fit0[0] = 100
    # fit0[1] = hi_mod.params[1]
    # fit0[2] = hi_mod.params[0]
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    # #gppmax *= gpp_optres.x[1]
    # pars = gpp_optres.x
    # gppmax = np.clip(df_to_fit.par,100,2000)**pars[1]  *  np.exp(pars[2])# np.exp(yearmat.dot(pars[3:]))

    # df_to_fit["gppmax"] = gppmax
    # kg = df_to_fit["gppmax"]/gpp_optres.x[0]
    # #%%
    # def gpp_opt(pars):
    #     #b_par, b_airt, b_airt2 = coefs[:3]
    #     #year_coef = coefs[3:]
    #     gppmax_pred = np.clip(df_to_fit.par,100,2000)**pars[1] *  np.exp(yearmat.dot(pars[2:]))
    #     gpp_pred = gppmax_pred*(1-np.exp(-g_samp*pars[0]/gppmax_pred))
    #     return (gpp_pred-gpp_samp)[np.isfinite(gpp_samp)]
    
    # fit0 = np.zeros(yearmat.shape[1]+2)
    # fit0[0] = 100
    # fit0[1] = hi_mod.params[-1]
    # fit0[2:] = hi_mod.params[0]
    # gpp_optres = scipy.optimize.least_squares(gpp_opt,x0=fit0,method="lm")#,x_scale=np.abs(fit0))
    # #gppmax *= gpp_optres.x[1]
    # pars = gpp_optres.x
    # gppmax = np.clip(df_to_fit.par,100,2000)**pars[1]  *  np.exp(yearmat.dot(pars[2:]))

    # df_to_fit["gppmax"] = gppmax
    # kg = df_to_fit["gppmax"]/gpp_optres.x[0]
    
    
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
    gpp_pred = gppmax*(1-np.exp(-(g_samp-res_cond)/kg))
    gpp_r2 = 1-np.nanmean((gpp_pred-gpp_samp)**2)/np.nanvar(gpp_samp)
    df_to_fit["gppR2"] = gpp_r2
    df_to_fit["gpp_pred"] = gpp_pred
    #gpp_r2_null = 1-np.mean((df_to_fit.gppmax-df_to_fit.gpp)**2)/np.var(df_to_fit.gpp)
    gpp_r2_null = np.corrcoef(gpp_samp[np.isfinite(gpp_samp)],gppmax[np.isfinite(gpp_samp)])[0,1]**2

    df_to_fit["gpp_slope"] = 100#gpp_optres.x[2]

    return df_to_fit, gpp_r2, gpp_r2_null
#%%
def prepare_df(fname, site_id, bif_forest):
    #%%
    df = pd.read_csv(fname,parse_dates=["TIMESTAMP"])
    df[df==-9999] = np.nan
    
    df["date"] = df["TIMESTAMP"].dt.date
    df["hour"] = df["TIMESTAMP"].dt.hour
    df["doy"] = df["TIMESTAMP"].dt.dayofyear

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
        #site_result[site_id] = "Not enough data"
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
    # #%%
    
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
        #site_result[site_id] = "Long-term ET exceeds P"
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
    # p_filter = 0*p_in + np.mean(p_in)
    # for i in range(365,len(p_filter)):
    #     p_filter[i] = np.mean(p_in[i-365:i])
    
    # rawsum = np.cumsum(p_in-et_out)
    # runoff = 0*rawsum + np.mean(p_in-et_out)
    # runoff[182:-183] = (rawsum[365:]-rawsum[:-365])/365
    # runoff = np.clip(runoff,0,np.inf)
    #%%
    #runoff = np.mean(p_in - et_out)
    #infil_frac = np.mean(et_out)/np.mean(p_in)
    wbi = 0
    waterbal_raw = np.zeros(len(doy_summer))
    for dayi in range(len(p_in)):
        waterbal_raw[dayi] = wbi
        wbi += p_in[dayi] - et_out[dayi] #+ runoff[dayi]
        wbi = min(0,wbi)
        # if dayi % 365 == summer_start:
        #     wbi = 0
    waterbal_corr = 1*waterbal_raw
    waterbal_corr[to_replace] = np.nan
    #%%
    # wbi = 0
    # aqi = 0
    # #%%
    # for runj in range(2):
    #     waterbal_raw = np.zeros(len(doy_summer))
    #     aquifer = np.zeros(len(doy_summer))
    #     #infil_frac = np.mean(et_out)/np.mean(p_in)
    #     for dayi in range(len(p_in)):
    #         waterbal_raw[dayi] = wbi
    #         aquifer[dayi] = aqi
    #         flowi = 5e-3*(aqi-wbi)
    #         wbi += -et_out[dayi] + flowi
    #         aqi += p_in[dayi] - flowi
    #         wbi = min(0,wbi)
            
        
    #     #wbi = min(0,wbi)
    #     # if dayi % 365 == summer_start:
    #     #     wbi = 0
    # waterbal_corr = 1*waterbal_raw
    # waterbal_corr[to_replace] = np.nan
    
    
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
    
    
    # smc_summer = np.array(meancols(df,'SWC'))
    # if np.sum(np.isfinite(smc_summer[np.isfinite(waterbal_corr)])) == 0:
    #     return "no soil moisture data"
    
    # sinterp = np.interp(smc_summer,np.sort(smc_summer[np.isfinite(waterbal_corr)]),
    #                     np.sort(waterbal_corr[np.isfinite(waterbal_corr)]))
    
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
    gpp_summer = (0*np.array(df["GPP_NT_VUT_REF"]) + 2*np.array(df["GPP_DT_VUT_REF"]))/2
    #airt_summer[airt_summer < 0] = np.nan
    gpp_summer[gpp_summer < 0] = np.nan
    
    nee_qc = np.array(df.NEE_VUT_REF_QC)
    gpp_summer[nee_qc < 0.5] = np.nan
    
    is_summer = (doy_summer >= summer_start)*(doy_summer <= summer_end)
    is_late_summer = (doy_summer >= topday)*(doy_summer <= summer_end)
    
    pet = petVnum/(sV+gammaV)
    
    
    # wb_transform = 0*waterbal_corr
    # for yr in pd.unique(df.year):
    #     sel = df.year==yr
    #     x = df.SWC_F_MDS_1[sel]*1
    #     y = waterbal_corr[sel]*1
    #     bg = np.isfinite(x*y)
    #     wb_transform[sel] = (y - np.mean(y[bg]))*np.std(x[bg])/np.std(y[bg]) + np.mean(x[bg])
    # wb_transform *= np.nanstd(waterbal_corr)/np.nanstd(wb_transform)
    # wb_transform -= np.nanmax(wb_transform)
    #%%
    if np.sum(np.isfinite(gpp_summer)) < (25):
        print("Not enough GPP")
        #site_result[site_id] = "Not enough data"

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
    #rain_fake = 1*rain_summer
    #rain_fake[doy_summer==summer_end] = np.inf
    #rain_for_dict = [rain_fake[is_late_summer],np.array(df.year)[is_late_summer]]

    rain_fake = 1*rain_summer
    rain_fake[doy_summer==summer_end] = np.inf
    rain_for_dict = [rain_fake[is_summer],np.array(df.year)[is_summer]]





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
                              #"smc":smc_summer
                              "vpd":vpd_summer,
                              "rnet":myrn,
                              "et_unc":df.LE_RANDUNC/44200,
                             # "waterbal_smc":wb_transform,
                              #"sinterp":sinterp,
                              #"gpp_unc":-df.NEE_VUT_REF_RANDUNC/df.NEE_VUT_REF,
                              "gpp_unc":df.NEE_VUT_REF_RANDUNC/gpp_summer,
                              "summer_start":summer_start,
                              "summer_peak":topday,
                              "summer_end":summer_end,
                              "summer_rain_freq":np.mean(rain_summer[is_summer]>0),
                              "late_summer_rain_freq":np.mean(rain_summer[is_late_summer]>0)
                              
                              #"PET":pet
                              })
                              #"LE_unc":df.LE_RANDUNC/44200})
    df_to_fit = df_to_fit_full.loc[is_summer].dropna()
    #df_to_fit = pd.merge(df_to_fit, df[["date","SWC_F_MDS_1"]])
    
    #df_all_summer = df_to_fit.copy()
    
    df_to_fit = df_to_fit.loc[df_to_fit.rain==0]
    df_to_fit = df_to_fit.loc[df_to_fit.rain_prev==0]

    df_to_fit = df_to_fit.loc[df_to_fit.doy >= topday].copy()
    df_to_fit = df_to_fit.loc[(df_to_fit.et_unc / df_to_fit.ET) <= 0.2].copy()
    
    
    #%%
    gpp_qc = np.array(df_to_fit.gpp)
    gpp_qc[(df_to_fit.gpp_unc <= 0) | (df_to_fit.gpp_unc > 0.2)] = np.nan
    #df_to_fit = df_to_fit.loc[(df_to_fit.gpp_unc >= 0)*(df_to_fit.gpp_unc < 0.15)].copy()
    df_to_fit["gpp_qc"] = gpp_qc
    
#%%
    year_count = df_to_fit.groupby("year").count()
    year_count["ETcount"] = year_count["ET"]*1
    df_to_fit = pd.merge(df_to_fit,year_count["ETcount"],on="year",how='left')
    df_to_fit = df_to_fit.loc[df_to_fit["ETcount"] >= 10].copy()

    return df_to_fit, rain_for_dict, np.nanmean(airt_summer), np.nanmean(rain_summer), df_to_fit_full
#, waterbal_raw, waterbal_corr

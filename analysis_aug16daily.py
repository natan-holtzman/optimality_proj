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
#import glob
import statsmodels.formula.api as smf

import matplotlib as mpl
#import h5py
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
mol_s_to_mm_day = 1*18/1000*24*60*60
#%%

rain_dict = {}
year_tau_dict = {}
site_result = {}

#%%
df_base = pd.read_csv("gs_67_mar23_evi.csv").groupby("SITE_ID").first().reset_index()

df_base["Aridity"] = df_base.mean_netrad / (df_base.map_data / (18/1000 * 60*60*24) * 44200)
df_base["Aridity_gs"] = df_base.gs_netrad / (df_base.mgsp_data / (18/1000 * 60*60*24) * 44200)

#%%


df_in = pd.read_csv("gs_67lai_climseas_april9.csv")#[["SITE_ID","EVI2","date"]]
rain_data = pd.read_csv("rain_67lai_climseas_april9.csv")

df_in = pd.merge(df_in,df_base[["SITE_ID","Aridity","Aridity_gs","mat_data","map_data"]],on="SITE_ID",how="left")

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
#bigyear = pd.read_csv("all_yearsites.csv")

#bigyear = pd.read_csv("hourly_gs_data_fullyear_parMM.csv")
#bigyear = pd.read_csv("hourly_gs_data_lai75d_parMM.csv")

#bigyear = pd.read_csv("hourly_gs_data_lai75_clim3_parMM.csv")
#bigyear = pd.read_csv("hourly_gs_data_lai50_clim_scale.csv")

bigyear = pd.read_csv("hourly_gs_data_lai75_clim_aug17.csv")

#bigyear = pd.read_csv("hourly_gs_data_lai75b_parMM.csv")

gppD = np.array(bigyear.gpp_dt+bigyear.gpp_nt)/2
gppD[np.array(bigyear.nee_qc)<0.5]= np.nan
bigyear["gpp"] = np.clip(gppD,0,np.inf)

# bigyear2 = pd.read_csv("AU_samplesites.csv")
# bigyear3 = pd.read_csv("US_samplesites.csv")

# bigyear = pd.concat([bigyear1,bigyear2,bigyear3]).reset_index()

bigyear = pd.merge(bigyear,bif_forest,on="SITE_ID",how='left')

bigyear["combined_biome"] = [simple_biomes[x] for x in bigyear["IGBP"]]

#bigyear = bigyear.loc[bigyear.year >= 2001].copy()
#%%
fullyear = pd.read_csv("all_yearsites_2gpp.csv")
#%%
site_message = []

#%%
all_results = []
ddlist = []
#for site_id in pd.unique(df_in.SITE_ID)[:]:#[forest_daily[x] for x in [70,76]]:
#for site_id in goodsites:#[forest_daily[x] for x in [70,76]]:
#for site_id in ["AU-DaS","AU-DaP","AU-How","US-Me2","US-MMS"]:
#for site_id in ["IT-Ro2","IT-SRo"]:
for site_id in pd.unique(bigyear.SITE_ID):
    #%%
    #if site_id=="ZM-Mon":
    #    continue
#%%
    print(site_id)
    dfgpp = df_in.loc[df_in.SITE_ID==site_id].copy()
    dfull = bigyear.loc[bigyear.SITE_ID==site_id].copy()
    dyear = fullyear.loc[fullyear.SITE_ID==site_id].copy()
    dfull.loc[dfull.gpp <= 0,"ET"] = np.nan
    #%%
    z1 = np.array(dyear.rain)
    y1 = np.array(dyear.year_new)
    L1 = np.array(dyear.LAI)
    GPP1 = np.clip(np.array(dyear.gpp_nt+dyear.gpp_dt),0,np.inf)

    rain_days = np.array([0] + list(np.where(z1 > 0)[0]) + [len(z1)])
    ddlenI = np.diff(rain_days)#-1
    ddlenY = y1[rain_days[:-1]]
    ymax = [np.max(ddlenI[ddlenY==y]) for y in pd.unique(ddlenY)]
    dfull["fullyear_dmax"] = np.mean(ymax)
    dfull["fullyear_dmean"] = np.sum(ddlenI**2)/np.sum(ddlenI)
    #%%
    if len(dfull) < 25:
        site_message.append("Not enough data")
        continue
    #dcount = dfull.groupby("year_new").count().reset_index()
    #fullyear = dcount.year_new.loc[dcount.combined_biome > 300]
   # dfull = dfull.loc[dfull.year_new.isin(fullyear)]
    #%%
    if np.max(dfull.LAI) < 0.05:
        site_message.append("No LAI data")
        continue
    
    #%%
    dfull["LAI"] = np.clip(dfull["LAI"],0.05,np.inf)

    dfull["is_summer"] = True #(dfull.doy >= summer_start)*(dfull.doy <= summer_end)

    dfull["dayfrac"] = 1-dfull.NIGHT
    #dfull["kgpp"] = dfull.gA_daily#*dfull.dayfrac
    dfull["vpd_fullday"] = 1*dfull.vpd
    dfull["vpd"] = np.clip(dfull.vpd_daytime,0.1,np.inf)
    #dfull["vpd"] = np.clip(dfull.vpd,0.1,np.inf)


    dfull["cond2"] = dfull.ET/np.clip(dfull.vpd_daytime,0.1,np.inf)*100

    #dfull["gpp_pred"] = dfull.amax_hourly*(1 - np.exp(-dfull.cond2/dfull.gA_hourly))
    #dfull["gpp_pred_const"] = dfull.amax_hourly*(1 - np.exp(-np.mean(dfull.cond2)/dfull.gA_hourly))

    #dfull["gpp_pred"] = dfull.amax_daily*(1 - np.exp(-dfull.cond2/dfull.gA_daily))
   # dfull["gpp_pred_const"] = dfull.amax_daily*(1 - np.exp(-np.mean(dfull.cond2)/dfull.gA_daily))

    #dfull["gpp_pred_lin"] = dfull.amax_daily*(dfull.cond2/dfull.gA_daily)

    #daymod = smf.ols("np.log(gpp/gpp_pred) ~ doy + np.power(doy,2)",data=dfull,missing='drop').fit()

    # dayfac = np.exp(daymod.predict(dfull))
    # dfull.kgpp *= dayfac
    # dfull["gpp_pred2"] = dayfac*dfull.amax_hourly*(1 - np.exp(-dfull.cond2/dfull.gA_hourly/dayfac))

    #dfull["gppR2_exp"] = r2_skipna(dfull.gpp_pred,dfull.gpp)
    #dfull["gppR2_const"] = r2_skipna(dfull.gpp_pred_const,dfull.gpp)
    #%%
    dfday = dfull[["par","LAI","cond2","gpp","vpd","rain","rain_prev"]].dropna()
    dfday = dfday.loc[dfday.rain==0].copy()
    dfday = dfday.loc[dfday.rain_prev==0].copy()

    # lowcut = -np.std(dfday.gpp/dfday.LAI)*2 + np.mean(dfday.gpp/dfday.LAI)
    # hicut = np.std(dfday.gpp/dfday.LAI)*2 + np.mean(dfday.gpp/dfday.LAI)
    # dfday = dfday.loc[dfday.gpp/dfday.LAI > lowcut]
    # dfday = dfday.loc[dfday.gpp/dfday.LAI < hicut].copy()

    #%%
    use_mm = 1
    if use_mm:
        def tofit(pars):
            amax1,kA,gmax1,kG = pars
            
            amax = amax1*dfday.par/(dfday.par + kA) * dfday.LAI
            gA = gmax1*dfday.par/(dfday.par + kG) * dfday.LAI
    
            gpp_pred = amax*(1-np.exp(-dfday.cond2/gA))
            z = (gpp_pred-dfday.gpp)#[dfday.VPD > 1]
            return z
        himean = np.quantile(dfday.gpp/dfday.LAI,0.9)
        fit0 = np.array([himean,600,himean/200,600])
        myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))

        amax1,kA,gmax1,kG = myfit.x
    #%%
    dfull["amax1"] = amax1
    dfull["kA1"] = kA
    dfull["gmax1"] = gmax1
    dfull["kG1"] = kG
    # bspar = []
    # bsmean = []
    # for bsi in range(20):
    #     dfday2 = dfday.sample(len(dfday),replace=True)
    #     def tofit(pars):
    #         amax1,kA,gmax1,kG = pars
            
    #         amax = amax1*dfday2.par/(dfday2.par + kA) * dfday2.LAI
    #         gA = gmax1*dfday2.par/(dfday2.par + kG) * dfday2.LAI
    
    #         gpp_pred = amax*(1-np.exp(-dfday2.cond2/gA))
    #         z = (gpp_pred-dfday2.gpp)#[dfday.VPD > 1]
    #         return z
    #     myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    #     amax1,kA,gmax1,kG = myfit.x
    #     bspar.append(myfit.x)
    #     gA = gmax1*dfull.par/(dfull.par + kG) * dfull.LAI
    #     bsmean.append(np.nanmean(gA))

        
    # #%%
    # amax1 = 15.0
    # def tofit(pars):
    #     kA,gmax1,kG = pars
        
    #     amax = amax1*dfday.par/(dfday.par + kA) * dfday.LAI
    #     gA = gmax1*dfday.par/(dfday.par + kG) * dfday.LAI

    #     gpp_pred = amax*(1-np.exp(-dfday.cond2/gA))
    #     z = (gpp_pred-dfday.gpp)[dfday.par > 100]
    #     return z
    # himean = np.quantile(dfday.gpp/dfday.LAI,0.9)
    # fit0 = np.array([300,amax1/200,300])
    # myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))

    # kA,gmax1,kG = myfit.x
    # #%%
    # def tofit(pars):
    #     kP,amax1,gmax1 = pars
        
    #     amax = amax1*dfday.par/(dfday.par + kP) * dfday.LAI
    #     gA = gmax1*dfday.par/(dfday.par + kP) * dfday.LAI

    #     gpp_pred = amax*(1-np.exp(-dfday.cond2/gA))
    #     z = (gpp_pred-dfday.gpp)[dfday.par > 100]
    #     return z
    # himean = np.quantile(dfday.gpp/dfday.LAI,0.9)
    # fit0 = np.array([200,6,6/200])
    # myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))

    # kP,amax1,gmax1 = myfit.x
    
    
    # #%%
    dfull["amax_d2"] = amax1*dfull.par/(dfull.par + kA) * dfull.LAI
    dfull["gA_d2"] = gmax1*dfull.par/(dfull.par + kG) * dfull.LAI

    dfull["gpp_pred_d2"] = dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2))
    
    dfull["kgpp"] = dfull.gA_d2#*dfull.dayfrac
    
    dfull["gpp_pred_d2a"] = 1.5*dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/1.5))
    dfull["gpp_pred_d2b"] = 0.5*dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/0.5))
    
    
    dfull["gpp_pred_d2c"] = dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/2))
    dfull["gpp_pred_d2d"] = dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/0.5))


    # df["gA_hourly"] = gmax1*df.PPFD_in/(df.PPFD_in + kG) * df.LAI
    # df["amax_hourly"] = amax1*df.PPFD_in/(df.PPFD_in + kA) * df.LAI
    # df["gpp_pred_hourly"] = df["amax_hourly"] * (1 - np.exp(-df.cond/df["gA_hourly"]))
    # #%%
    #%%
    
    use_power = 0
    
    if use_power:
    
        gmax_par = []
        parbins = np.quantile(dfday.par,np.linspace(0,1,11))
        parmids = 0.5*(parbins[:-1] + parbins[1:])
        for i in range(10):
            dsel = dfday.loc[(dfday.par >= parbins[i])*(dfday.par < parbins[i+1])]
            gmax_par.append(np.quantile(dsel.gpp/dsel.LAI,0.9))
        #%%
        #dfday["amax_interp"] = np.interp(dfday.par,parmids,gmax_par)
        parmod = sm.OLS(np.log(gmax_par),sm.add_constant(np.log(parmids))).fit()
        
        dfday["amax_interp"] = dfday.par**parmod.params[1] * np.exp(parmod.params[0])
    
        #%%
        def tofit(pars):
            amax,k2 = pars
    
            gpp_pred = amax*(1-np.exp(-dfday.cond2/dfday.LAI/dfday.amax_interp / k2)) *dfday.LAI*dfday.amax_interp
            z = (gpp_pred-dfday.gpp)#[dfday.par > 100]
            return z
        fit0 = np.array([0.9,0.9/150])
        myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    
        amax,k2 = myfit.x
        #%%
        dfull["amax_interp"] = dfull.par**parmod.params[1] * np.exp(parmod.params[0])
        
        dfull["amax_d2"] = amax*dfull.LAI*dfull.amax_interp
        dfull["gA_d2"] = k2*dfull.LAI*dfull.amax_interp
        dfull["gpp_pred_d2"] = dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2))
        
        dfull["kgpp"] = dfull.gA_d2
    #%%
    gAtest = gmax1*dfday.par/(dfday.par + kG) * dfday.LAI
    z1 = 1-np.exp(-dfday.cond2/gAtest)
    dfull["frac_gt9"] = np.mean(z1 > 0.9)
    
    residG = np.log(dfull.gpp/dfull.gpp_pred_d2)
    residcors = []
    for var1 in ["airt","doy","LAI","par","smc"]:
        try:
            residcors.append(cor_skipna(dfull[var1],residG)[0])
        except:
            pass
    dfull["max_resid_cor"] = np.max(np.abs(np.array(residcors)))
    
    # afrac = 1 - np.exp(-dfull.cond2/dfull.gA_daily)
    # frac_above_9 = np.sum(afrac > 0.9)/np.sum(np.isfinite(afrac))
    # if frac_above_9 < 0.1:
    #     site_message.append("Not enough hi GPP")
    #     continue
    # if frac_above_9 > 0.9:
    #     site_message.append("Not enough lo GPP")
    #     continue
    #%%
    dfull["gppR2_exp"] = r2_skipna(dfull.gpp_pred_d2,dfull.gpp)
    if r2_skipna(dfull.gpp_pred_d2,dfull.gpp) < 0:
        site_message.append("GPP model did not fit")
        continue
    
    #smin_mm = -500
    #tauDay = 50
    #%%
    dfGS = dfull.loc[dfull.is_summer].copy()
    dfull["gsrain_mean"] = np.mean(dfGS.rain)
#    dfGS = dfull.copy()
    #dfGS["cond_per_LAI"] = dfGS.cond/dfGS.LAI
    #cl75 =  np.nanquantile(dfGS["cond_per_LAI"],0.75)
#%%
    seaslens = []
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
    ddyears = []
    
    ddall = 0
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
        vpd_arr = np.clip(np.array(dfy.vpd),0.1,np.inf)/100
        if np.sum(np.isfinite(vpd_arr)) < 25:
            continue
        vpd_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(vpd_arr)],
                            vpd_arr[np.isfinite(vpd_arr)])
        k_mm_day = np.array(dfy.kgpp)*mol_s_to_mm_day #* np.array(dfy.gpp/dfy.gpp_pred)
        rain_arr = np.array(dfy.rain)
        seaslens.append(len(rain_arr))
        #%%
    #et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        
        et_mmday = np.array(dfy.ET)*mol_s_to_mm_day
        
        #et_mmday = np.array(dfy.cond)*mol_s_to_mm_day
        
        et_mmday_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(et_mmday)],
                            et_mmday[np.isfinite(et_mmday)])
        # w_arr = np.array(dfy.waterbal)
        
        #cond1 = np.array(dfy.cond_per_LAI)
        #et_mmday[cond1 > cl75] = np.nan
        
        #et_mmday[dfy.airt < 10] = np.nan
        et_mmday[dfy.par < 100] = np.nan

        et_mmday[dfy.vpd <= 0.5] = np.nan
        
        #et_mmday[dfy.ET_qc < 0.5] = np.nan
        
#%%
        rain_d5 = np.array([0] + list(np.where(rain_arr > 5)[0]) + [len(rain_arr)])
        dd5= np.diff(rain_d5)
        ymaxes.append(np.max(dd5))
#        ymeans.append(np.mean(dd5[dd5 >= 2]))
        ymeans.append(np.sum(dd5**2)/np.sum(dd5))

#%%
        rain_days = np.array([0] + list(np.where(rain_arr > 0)[0]) + [len(rain_arr)])
        ddgood = np.where(np.diff(rain_days) >= 7)[0]
        
        ddall += len(ddgood)
        
        ddstart = rain_days[ddgood]+2
        ddend = rain_days[ddgood+1]
    
        etnorm = et_mmday**2 / (vpd_interp*k_mm_day)
        #etnorm[vpd_arr < 0.1/100] = np.nan

        
        #etnorm[dfy.airt < 10] = np.nan
        #etnorm[dfy.par < 100] = np.nan

        #etnorm[dfy.vpd < 0.5] = np.nan
        #etnorm[dfy.ET_qc < 0.5] = np.nan
        
        doyY = np.array(dfy.doy)
    #%%
        rain_days = np.array([0] + list(np.where(rain_arr > 0)[0]) + [len(rain_arr)])

        dd0= np.diff(rain_days) #- 1
        ymaxes0.append(np.max(dd0))
        #ymeans0.append(np.mean(dd0[dd0 >= 2]))
        ymeans0.append(np.sum(dd0**2)/np.sum(dd0))

    #%%
        #tau_with_unc = []
        #winit_with_unc = []
        for ddi in range(len(ddstart)):
            starti = ddstart[ddi]
            #starti = max(ddstart[ddi],ddend[ddi] - 50) 

            endi = ddend[ddi]
            #endi = min(starti+20,ddend[ddi])
            f_of_t = (vpd_arr*k_mm_day)[starti:endi]
#           # g_of_t = np.cumsum(np.sqrt(f_of_t))
#            g_of_t = np.array([0] + list(np.cumsum(np.sqrt(f_of_t))))[:-1]
            g_of_t = np.array([0] + list(np.cumsum(np.sqrt(f_of_t))))[:-1]
            #g_of_t = g_of_t[:20]
            #yfull = et_mmday[ddstart[ddi]:ddend[ddi]]/np.sqrt(f_of_t)
            #yfull = yfull[:20]
            
            doyDD = doyY[starti:endi]
            yfull = etnorm[starti:endi]#[:20]
            etsel = et_mmday_interp[starti:endi]#[:20]
            rainsel =  rain_arr[starti:endi]
            #if r1.params[1] < 0 and r1.pvalues[1] < 0.05:
            if np.sum(np.isfinite(yfull)) >= 5: # and np.mean(np.isfinite(yfull)) >= 0.75:
                #et_over_dd.append(yfull - np.nanmean(yfull))
                #ddreg_fixed.append(g_of_t - np.mean(g_of_t[np.isfinite(yfull)]))
                etcumDD = np.array([0] + list(np.cumsum(etsel-rainsel)))[:-1]

                rDD = sm.OLS(yfull,sm.add_constant(etcumDD),missing='drop').fit()
#                rDD = sm.OLS(yfull,sm.add_constant(g_of_t),missing='drop').fit()

#                if rDD.pvalues[1] < 0.1 and rDD.params[1] < 0:
                if rDD.rsquared > 0.25 and rDD.params[1] < 0:
                                       
                    
                    ddlabel.append([ddii]*len(yfull))
                    ddyears.append([y0]*len(yfull))
    
                    
                    frec.append(f_of_t)
                    grec.append(g_of_t)
                    vpd_plain.append(vpd_arr[starti:endi])
                    et_plain.append(et_mmday[starti:endi])
                    
                    etcum.append(etcumDD)
                    
                    et_over_dd.append(yfull - np.nanmean(yfull))
                    ddreg_fixed.append(etcumDD - np.mean(etcumDD[np.isfinite(yfull)]))
                    
                    #et_over_dd.append((yfull - np.nanmean(yfull))/np.std(etcumDD))
                    #ddreg_fixed.append((etcumDD - np.mean(etcumDD[np.isfinite(yfull)]))/np.std(etcumDD))
    
                    ddii += 1
        #%%
        
    #%%
    if ddall < 3:
        site_message.append("Not enough dd")
        continue
    #%% 
    if len(ddreg_fixed) < 3:
        site_message.append("Not enough water limitation")
        continue
#     #%%
#%%    
    row0 = np.concatenate(ddreg_fixed)
    #row0[np.abs(row0) > np.nanstd(row0)*3] = np.nan

    et_topred = np.concatenate(et_over_dd)
    et_topred[np.abs(et_topred) > np.nanstd(et_topred)*3] = np.nan
    #et_topred[np.abs(row0) > np.nanstd(row0[np.isfinite(et_topred)])*3] = np.nan

    # if np.sum(np.isfinite(et_topred*row0)) < 10:
    #     continue
    #%%
    r1= sm.OLS(et_topred,row0,missing='drop').fit()
    # if r1.pvalues[0] > 0.05 or r1.params[0] > 0:
    #     continue
    dfull["reg_npoints"] = np.sum(np.isfinite(et_topred))
    dfull["reg_ndd"] = len(et_over_dd)
    dfull["reg_pval"] = r1.pvalues[0]
    #%%
    dfull["tau_ddreg"] = -2/r1.params[0]
    dfull["tau_ddreg_lo"] = -2/(r1.params[0] - 2*r1.bse[0])
    dfull["tau_ddreg_hi"] = -2/(r1.params[0] + 2*r1.bse[0])
    #%%
    dfull["gslen_annual"] = np.mean(seaslens)
    dfull["tau_rel_err"] = -r1.bse[0]/r1.params[0]
    #dfull = dfull.loc[dfull.year_new.isin(pd.unique(dfgpp0.year_new))].copy()
    #dfull["dayfrac"] = (1-dfull.NIGHT)
    dfull["seas_rain_mean5"] = np.mean(ymeans)
    dfull["seas_rain_max5"] = np.mean(ymaxes)
    dfull["seas_rain_mean0"] = np.mean(ymeans0)
    dfull["seas_rain_max0"] = np.mean(ymaxes0)
    #%%
    btab = pd.DataFrame({"SITE_ID":site_id,
        "ddi":np.concatenate(ddlabel),
                         "G":np.concatenate(grec),
                         "ET":np.concatenate(et_plain),
                         "et_per_F_dm":et_topred,
                         "row0":row0,
                         "F":np.sqrt(np.concatenate(frec)),
                         "VPD":np.concatenate(vpd_plain),
                         "etcum":np.concatenate(etcum),
                         "year":np.concatenate(ddyears),
                         "ddlen":np.concatenate([[len(x)]*len(x) for x in vpd_plain])})
    #%%
    btab["cond"] = btab.ET/btab.VPD
    tau = -2/r1.params[0]
    #%%
    # looval = []
    # for ddz in pd.unique(btab.ddi):
    #     btab2 = btab.loc[btab.ddi != ddz].copy()
    #     looval.append(-2/sm.OLS(btab2.et_per_F_dm, btab2.row0, missing='drop').fit().params[0])
    #%%
    # btab["ET2"] = btab.ET**2
    # breg = smf.ols("ET2 ~ 0 + ")
    
    #btab["term1"] = -2/tau*btab.etcum
    #%%
#     ddmean = btab.groupby("ddi").mean().reset_index()
#     ddmean["b"] = (ddmean.ET - ddmean.term1)/ddmean.F
#     #%%
#     btab = pd.merge(btab,ddmean[["ddi","b"]],on='ddi',how='left')
# #%%
#     btab["ETpred"] = btab.term1 + btab.b*btab.F
#     dfull["etr2_b"] = r2_skipna(btab.ETpred,btab.ET)
#     dfull["gr2_b"] = r2_skipna(btab.ETpred/btab.VPD,btab.cond)
#     #%%
#     bmod = smf.ols("ET ~ 0 + G:F + F:C(ddi)",data=btab,missing='drop').fit()
#     bmod0 = smf.ols("ET ~ 0 + F:C(ddi)",data=btab,missing='drop').fit()
#%%
    btab["etnorm"] = btab.ET**2/btab.F**2
    btab["et2"] = btab.ET**2
    btab["F2"] = btab.F**2
    #%%
    #cmod = smf.ols("etnorm ~ 0 + etcum + C(ddi)",data=btab,missing='drop').fit()
    #cmod0 = smf.ols("etnorm ~ 0 + C(ddi)",data=btab,missing='drop').fit()
    #%%
    dmod = smf.ols("et2 ~ 0 + etcum:F2 + C(ddi):F2",data=btab,missing='drop').fit()
    dmod0 = smf.ols("et2 ~ 0 + C(ddi):F2",data=btab,missing='drop').fit()
    dmod2 = smf.ols("et2 ~ 0 + etcum:F2 + np.power(etcum,2):F2 + C(ddi):F2",data=btab,missing='drop').fit()

    #%%
    # cutoff = 300
    # btab3 = btab2.loc[btab2.cond < cutoff].copy()
    # dmod = smf.ols("et2 ~ 0 + etcum:F2 + C(ddi):F2",data=btab3,missing='drop').fit()
    # ddpresent = btab2.ddi.isin(pd.unique(btab3.ddi))
    # etmax = btab2.VPD*cquant[j]
    # dpred = 1*etmax
    # dpred[ddpresent] = np.sqrt(dmod.predict(btab2.loc[ddpresent]))
    # dpred = np.clip(dpred,0,etmax)
    # mses.append(np.nanmean(np.abs(dpred-btab2.ET)))
    
    #%%
    ddlist.append(btab)
    # dfull["etnorm_r2"] = cmod.rsquared
    # dfull["etnorm_r2_0"] = cmod0.rsquared
    # dfull["et2_r2"] = dmod.rsquared
    # dfull["et2_r2_0"] = dmod0.rsquared
#%%#%%
    # ETreg_pred = np.sqrt(np.clip(dmod.predict(btab),0,np.inf))
    # Greg_pred = ETreg_pred / btab.VPD
    
    # dfull["etr2_b"] = r2_skipna(ETreg_pred,btab.ET)
    # dfull["gr2_b"] = r2_skipna(Greg_pred,btab.cond)
    #%%
    
    #tab1dd = tab1.groupby("ddi").mean(numeric_only=True).reset_index()
    tab1first = btab.groupby("ddi").first().reset_index()
    
    tab1first["et_init"] = 1*tab1first.ET
    tab1first["g_init"] = 1*tab1first.cond

    tab2 = pd.merge(btab,tab1first[["ddi","et_init","g_init"]],how="left",on="ddi")


    tab2["mydiff"] = tab2.et2*tau/2 + tab2.etcum*tab2.F2
    dmod2 = smf.ols("mydiff ~ 0 + C(ddi):F2",data=tab2,missing='drop').fit()
    epredN = np.sqrt(np.clip(dmod2.predict(tab2)*2/tau - tab2.etcum*tab2.F2*2/tau,0,np.inf))

    #dmod = smf.ols("et2 ~ 0 + etcum:F2 + C(ddi):F2",data=tab2,missing='drop').fit()
    #epredM = np.sqrt(np.clip(dmod.predict(tab2),0,np.inf))
    btab["etpred"] = epredN
    dfull["etr2_norm"] = r2_skipna(epredN/tab2.et_init,tab2.ET/tab2.et_init)
    dfull["gr2_norm"] = r2_skipna(epredN/tab2.VPD/tab2.g_init,tab2.cond/tab2.g_init)
    dfull["tau_simult"] = -2/dmod.params[0]
    #dfull
    # ddfirst = btab.groupby("ddi").first().reset_index()
    # #ddfirst["aETinit"] = 1*ddfirst["ET"]/ddfirst["F"]
    # ddfirst["ETinit"] = 1*ddfirst["ET"]
    # ddfirst["Finit"] = 1*ddfirst["F"]
    # btab = pd.merge(btab,ddfirst[["ddi","ETinit","Finit"]],on='ddi',how='left')
    # #%%
    # btab["b_init"] = (btab.ETinit/btab.Finit)
#%%tau = -2/r1.params[0]
    # etnorm = (dfull.ET*mol_s_to_mm_day)**2 / (dfull.vpd/100) / (dfull.gA_daily*mol_s_to_mm_day)
    # etnorm[dfull.par < 100] = np.nan
    # etnorm[dfull.vpd < 0.5] = np.nan
    # etnorm[dfull.rain > 0] = np.nan
    # etnorm[dfull.rain_prev > 0] = np.nan
    # s_inv = etnorm*tau/2
    # s_inv[np.abs(s_inv - np.nanmean(s_inv)) > 3*np.nanstd(s_inv)] = np.nan
    # #%%
    # x = np.array(dfull.smc)
    # y = np.array(s_inv)
    # x1 = x[np.isfinite(x*y)]
    # y1 = y[np.isfinite(x*y)]
    # cinterp = np.interp(x, np.sort(x1),np.sort(y1))
    # #%%
    # etpredS = np.sqrt(2/tau*cinterp*(dfull.vpd/100)* (dfull.gA_daily*mol_s_to_mm_day))
    # etpredS[np.isnan(etnorm)] = np.nan
    #%%
    #s_inv2 = nanterp(np.array(s_inv))
    # dfgpp0["etpred_newmod_mmday"] = etrec
    # dfi = pd.merge(dfi,dfgpp0[["date","etpred_newmod_mmday"]],on="date",how="left")
    # dfi["etr2_reg"] = 1- np.nanmean((dfgpp0.ET-dfgpp0.etpred_newmod_mmday/mol_s_to_mm_day)**2)/np.nanvar(dfgpp0.ET)
    # dfi["gr2_reg"] = 1- np.nanmean((dfgpp0.cond-dfgpp0.etpred_newmod_mmday/mol_s_to_mm_day/(dfgpp0.vpd/100))**2)/np.nanvar(dfgpp0.cond)
#%%
    all_results.append(dfull)
    
    site_message.append("Tau estimated")
    #%%
    # plt.figure(figsize=(14,10))
    # plt.subplot(2,3,1)
    # plt.plot(dfull.ET)
    # plt.title("ET")
    # plt.subplot(2,3,2)
    # plt.plot(dfull.vpd)
    # plt.title("VPD")
    # plt.subplot(2,3,3)
    # plt.plot(dfull.gpp)
    # plt.title("GPP")
    # plt.subplot(2,3,4)
    # plt.plot(dfull.gA_daily)
    # plt.title("gA")
    # plt.subplot(2,3,5)
    # plt.plot(dfull.cond)
    # plt.title("g")
    # plt.subplot(2,3,6)
    # plt.plot(dfull.LAI)
    # plt.title("LAI")
    # plt.suptitle(site_id)
    # #%%
    # plt.figure()
    # plt.plot(row0,et_topred,'o')
    # plt.title(site_id)
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
df1b = df1.loc[df1.gppR2_exp > 0].copy()

#%%
#%%
df_meta= df1b.loc[df1b.tau_ddreg > 0]
#df_meta= df_meta.loc[df_meta.reg_pval < 0.05]

#fval = ((1-df_meta.etr2_null)-(1-df_meta.etr2_smc))/(1-df_meta.etr2_smc)*(df_meta.npoints-4)
#df_meta["ftest"] = 1-scipy.stats.f.cdf(x=fval,dfn=1,dfd=df_meta.npoints-4)
#df_meta = df_meta.loc[df_meta.ftest < 0.01]
#df_meta = df_meta.loc[df_meta.LOCATION_LAT > 0]
#df_meta = df_meta.loc[df_meta.tau_rel_unc < 0.25].copy()
#udf = all_results.groupby("SITE_ID").nunique().reset_index()
#udf["nyears"] = 1*udf["year"]
#df_meta = pd.merge(df_meta,udf[["SITE_ID","nyears"]],on="SITE_ID",how="left")
#%%
df_meta = pd.merge(df_meta,metadata,left_on="SITE_ID",right_on="fluxnetid",how="left")
#%%
#df_meta = df_meta.loc[df_meta["cor_gpp_pval"] < 0.05]
#%%
#df_meta = df_meta.loc[df_meta.gppR2_exp - df_meta.gppR2_lin > 0.01]
#df_meta = df_meta.loc[df_meta.gppR2_exp - df_meta.gppR2_linC > 0.01]
#%%
#df_meta = df_meta.loc[df_meta.gppR2_base - df_meta.gppR2_only_cond > 0.05]
#df_meta = df_meta.loc[df_meta.gppR2_base - df_meta.gppR2_no_cond > 0.05]
#df_meta = df_meta.loc[df_meta.inflow == 0]
df_meta = df_meta.loc[df_meta.tau_ddreg_lo > 0]
df_meta = df_meta.loc[df_meta.tau_ddreg_hi > 0]
#%%
#df_meta = df_meta.loc[(df_meta.tau_hi-df_meta.tau_lo)/df_meta.tau_reg < 0.75]
#df_meta = df_meta.loc[df_meta.tau_rel_err < 0.25]
#df_meta = df_meta.loc[df_meta.gpp_par_err_rel < 0.05]
#df_meta = df_meta.loc[(df_meta.tau_ddreg_hi-df_meta.tau_ddreg_lo)/ df_meta.tau_ddreg < 1].copy()
#df_meta = df_meta.loc[df_meta.tau_ddreg_hi-df_meta.tau_ddreg_lo < 25].copy()
df_meta = df_meta.loc[df_meta.reg_ndd >= 3].copy()

#%%
df_meta["ddrain_mean"] = 1*df_meta.seas_rain_max0
df_meta["ddrain_2mean"] = 1*df_meta.seas_rain_mean0
#%%
df_meta["gsrain_len"] = df_meta.gslen_annual
#df_meta.summer_end - df_meta.summer_start
#%%
df_meta = df_meta.loc[df_meta.frac_gt9 > 0.01].copy()
df_meta = df_meta.loc[df_meta.frac_gt9 < 0.99].copy()
#df_meta = df_meta.loc[df_meta.max_resid_cor < 0.5].copy()
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
ax.set_xlabel("Growing season $D_{max}$ (days)",fontsize=24)
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
ax.set_xlabel("Average temperature ($^oC)$",fontsize=24)
ax.set_ylabel("Average annual precip. (cm)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%


#%%
df_meta3 = df_meta.sort_values("etr2_norm")
df_meta3["et_rank"] = np.arange(len(df_meta3))

fig,axes = plt.subplots(3,1,figsize=(16,10))
ax = axes[1]

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.et_rank,subI.etr2_norm,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
ax.set_xticks(df_meta3.et_rank,df_meta3.SITE_ID,rotation=90)
#ax.set_xlim(0,250)
ax.set_ylim(0,1)
#ax.set_xlabel("Rank",fontsize=24)
ax.set_title(r"$R^2$ of $ET/ET_{0}$ during water-limited drydowns",fontsize=24)

#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

df_meta3 = df_meta.sort_values("gr2_norm")
df_meta3["g_rank"] = np.arange(len(df_meta3))
ax = axes[2]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.g_rank,subI.gr2_norm,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.g_rank,df_meta3.SITE_ID,rotation=90)
ax.set_title(r"$R^2$ of $g/g_{0}$ during water-limited drydowns",fontsize=24)
#ax.axhline(0,color='k')


df_meta3 = df_meta.sort_values("gppR2_exp")
df_meta3["gpp_rank"] = np.arange(len(df_meta3))
ax = axes[0]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.gpp_rank,subI.gppR2_exp,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.gpp_rank,df_meta3.SITE_ID,rotation=90)
ax.set_title(r"$R^2$ of GPP given observed g during growing season",fontsize=24)
fig.tight_layout()
fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.02),ncols=2)
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%

#%%
biome_index = dict(zip(biome_list,range(len(biome_list))))
df_meta["biome_number"] = [biome_index[x] for x in df_meta.combined_biome]
#%%
plot_colors = mpl.colormaps["tab10"](df_meta["biome_number"] +2)
df_meta["tau"] = 1*df_meta.tau_ddreg
#%%
def myplot(ax,x,y,xlab,ylab):
    ax.scatter(x,y,c=plot_colors)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    myr2 = np.corrcoef(x,y)[0,1]**2
    ax.text(0.1,0.8,  "$R^2$ = " + str(np.round(myr2,2)),transform=ax.transAxes)
#%%
fig,axes=plt.subplots(3,3,figsize=(12,10))

myplot(axes[0,0],df_meta.Aridity,df_meta.tau,
       "Annual aridity index",r"$\tau$ (days)")

myplot(axes[0,1],df_meta.Aridity_gs,df_meta.tau,
       "GS aridity index","")

myplot(axes[1,0],df_meta.map_data,df_meta.tau,
       "Annual P (mm/day)",r"$\tau$ (days)")

myplot(axes[1,1],df_meta.gsrain_mean,df_meta.tau,
       "GS P (mm/day)","")

axes[0,2].set_axis_off()

myplot(axes[2,2],df_meta.ddrain_2mean,df_meta.tau,
       "GS $D_{mean}$ (days)","")

myplot(axes[1,2],df_meta.gsrain_len,df_meta.tau,
       "GS length (days)","")


myplot(axes[2,1],df_meta.fullyear_dmax,df_meta.tau,
       "Annual $D_{max}$ (days)","")


myplot(axes[2,0],df_meta.fullyear_dmean,df_meta.tau,
       "Annual $D_{mean}$ (days)",r"$\tau$ (days)")

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
#all_results = pd.merge(all_results,rain_site_tab,on="SITE_ID",how="left")
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
ddlist = pd.concat(ddlist)
#%%
#tabS = ddlist.loc[ddlist.SITE_ID=="US-Me5"]
site_pair = ["US-Me5","US-SRM"]
plt.figure()
si = 1
for x in site_pair:
    #plt.subplot(2,1,si)
    tabS = ddlist.loc[ddlist.SITE_ID==x].copy()
    tabS = tabS.loc[tabS.year >= 2001].copy()
    ddlens = tabS.groupby("ddi").count().reset_index()
    #longDD = np.argmax(ddlens.et_per_F_dm)
    longDD = np.argmin(np.abs(ddlens.et_per_F_dm-20))
    
    jtab = tabS[tabS.ddi==longDD].reset_index()
    istart = np.where(np.isfinite(jtab.ET))[0][0]
    jtab = jtab.iloc[istart:].copy()
    tau = 20
    #sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    #epred20 = np.sqrt(2/tau*jtab.F2*(sm_init-jtab.etcum))
    term1 = -1/tau*jtab.F*jtab.G
    #c2 = (np.nanmean(jtab.ET) - np.nanmean(term1))/np.nanmean(jtab.F)
    c2 = jtab.ET.iloc[0]/jtab.F.iloc[0]
    # sm0 = 10
    # c1 = np.sqrt(sm0*4)
    # c2 = 0.5*c1*np.sqrt(2/tau)
    epred20 = np.clip(term1 + c2*jtab.F,0,np.inf)
    
    # sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    # smlist = []
    # etlist = []
    # s = 1*sm_init
    # for i in range(len(jtab)):
    #     smlist.append(s)
    #     eti = min(s,np.sqrt(2/tau*s*jtab.F2.iloc[i]))
    #     etlist.append(eti)
    #     s -= eti
    
    # epred20 = etlist
    
    #sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    #c1 = np.sqrt(sm_init*4)
    #sm_pred = 0.25*(-np.sqrt(2/tau)*jtab.G + c1)**2
    #epred20 = -np.diff(sm_pred)
    
    tau = 50
    term1 = -1/tau*jtab.F*jtab.G
    #c2 = (np.nanmean(ej) - np.nanmean(term1))/np.nanmean(f2)
    #c2 = (np.nanmean(jtab.ET) - np.nanmean(term1))/np.nanmean(jtab.F)

    c2 = jtab.ET.iloc[0]/jtab.F.iloc[0]
    #c2 = 0.5*c1*np.sqrt(2/tau)
    # sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    # c1 = np.sqrt(sm_init*4)
    # sm_pred = 0.25*(-np.sqrt(2/tau)*jtab.G + c1)**2
    # epred50 = -np.diff(sm_pred)
    
    epred50 =  np.clip(term1 + c2*jtab.F,0,np.inf)
    
    
    # sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    # smlist = []
    # etlist = []
    # s = 1*sm_init
    # for i in range(len(jtab)):
    #     smlist.append(s)
    #     eti = min(s,np.sqrt(2/tau*s*jtab.F2.iloc[i]))
    #     etlist.append(eti)
    #     s -= eti
    
    # epred50 = etlist
    
    
    #plt.subplot(2,1,si)
    xvar = np.arange(len(jtab.ET))+2
    plt.plot(xvar, np.array(jtab.ET),'ko-',linewidth=3,label="Eddy covariance")
    plt.plot(xvar,epred50,'o-',color="tab:blue",linewidth=3,alpha=0.6,label=r"Model, $\tau$ = 50 days")
    plt.plot(xvar,epred20,'o-',color="tab:orange",linewidth=3,alpha=0.6,label=r"Model, $\tau$ = 20 days")
    if si == 1:
        #plt.ylabel("ET (mm/day)",fontsize=22)
        #plt.ylim(-0.1,2)
        plt.legend(fontsize=16)
    # if si == 2:
        

    #     plt.xlabel("Day of drydown",fontsize=22)
    #     plt.ylabel("ET (mm/day)",fontsize=22)
    #plt.title(x)
    si += 1
#plt.tight_layout()
#plt.ylim(-0.1,3.9)
plt.xticks(np.arange(2,23,3))
plt.xlabel("Day of drydown",fontsize=22)
plt.ylabel("ET (mm/day)",fontsize=22)
plt.text(1.5,1.25,site_pair[0],fontsize=20)
plt.text(1.5,2.5,site_pair[1],fontsize=20)
plt.ylim(0,2.75)

# sinf = (ej / np.sqrt(fj) / np.sqrt(2/tau))**2
# s = sinf[0]
# sirec = np.zeros(len(fj))
# for i in range(len(fj)):
#     sirec[i] = s
#     eti = np.sqrt(2/tau*s*fj[i])
#     s -= eti
# etrec = np.sqrt(sirec * 2/tau * fj)
#%%

#%%
plt.figure(figsize=(10,8))
plt.axvline(0,color="grey",linestyle="--")
plt.axhline(0,color="grey",linestyle="--")

tab1 =  ddlist.loc[ddlist.SITE_ID=="US-Me5"].copy()
tab2 =  ddlist.loc[ddlist.SITE_ID=="US-SRM"].copy()
#tab2 =  ddlist.loc[ddlist.SITE_ID=="US-ARc"].copy()

plt.plot(tab1.row0,tab1.et_per_F_dm,'o',label=r"US-Me5, $\tau$ = 43 days",alpha=0.6)
plt.plot(tab2.row0,tab2.et_per_F_dm,'o',label=r"US-SRM, $\tau$ = 16 days",alpha=0.6)
rA = sm.OLS(tab1.et_per_F_dm,tab1.row0,missing='drop').fit()
rB = sm.OLS(tab2.et_per_F_dm,tab2.row0,missing='drop').fit()
xarr = np.array([-25,25])
plt.plot(xarr,xarr*rA.params[0],color="tab:blue")
xarr = np.array([-15,15])
plt.plot(xarr,xarr*rB.params[0],color="tab:orange")

plt.xlabel("Cumulative ET, daily value minus drydown mean (mm)")
plt.ylabel("$ET_{norm}$ = $ET^2/(VPD*g_A*LAI)$,\ndaily value minus drydown mean (mm/day)")
plt.legend(loc="lower left")
#%%

# ddlist["dd_id"] = ddlist['SITE_ID'] + ddlist["ddi"].astype(str)
# ddlist = pd.merge(ddlist,df_meta[["SITE_ID","ddrain_mean"]],on="SITE_ID",how="left")
# ddlist["slopefac"] = 1/ddlist["ddrain_mean"]
# #ddlist["slopefac_ddi"] = 1/ddlist["ddlen"]

# m0 = smf.ols("etnorm ~ 0 + C(dd_id)",data=ddlist,missing='drop').fit()
# mSite = smf.ols("etnorm ~ etcum:C(SITE_ID) +  C(dd_id)",data=ddlist,missing='drop').fit()
# mSiteRain = smf.ols("etnorm ~ etcum:slopefac +  C(dd_id)",data=ddlist,missing='drop').fit()

# # mLen = smf.ols("etnorm ~ etcum + etcum:slopefac_ddi +  dd_id",data=ddlist,missing='drop').fit()
# #%%
# mSite2 = smf.ols("etnorm ~ etcum:C(SITE_ID) + np.power(etcum,2):C(SITE_ID) +  C(dd_id)",data=ddlist,missing='drop').fit()
# mSite2rain = smf.ols("etnorm ~ etcum:slopefac + np.power(etcum,2):slopefac +  C(dd_id)",data=ddlist,missing='drop').fit()

#sSiteLen = smf.ols("etnorm ~ etcum:slopefac_ddi + etcum:C(SITE_ID) +  dd_id",data=ddlist,missing='drop').fit()
#mSiteLen2 = smf.ols("etnorm ~ etcum:ddlen:C(SITE_ID) + etcum:C(SITE_ID) +  dd_id",data=ddlist,missing='drop').fit()

#mDDlen = smf.ols("etnorm ~ etcum:ddlen +  C(SITE_ID):C(ddi)",data=ddlist,missing='drop').fit()
#%%
# site_id = "US-Me2"
# tab1 =  ddlist.loc[ddlist.SITE_ID==site_id].copy()
# tau = float(df_meta.loc[df_meta.SITE_ID==site_id].tau_ddreg.iloc[0])
# tab1dd = tab1.groupby("ddi").mean(numeric_only=True).reset_index()
# tab1first = tab1.groupby("ddi").first().reset_index()
# tab1first["et_init"] = 1*tab1first.ET
# tab1first["f_init"] = 1*tab1first.F

# tab2 = pd.merge(tab1,tab1first[["ddi","et_init","f_init"]],how="left",on="ddi")

# dmod = smf.ols("et2 ~ 0 + etcum:F2 + C(ddi):F2",data=tab2,missing='drop').fit()
# epredM = np.sqrt(np.clip(dmod.predict(tab2),0,np.inf))
### tau = 31.8
# tab2["mydiff"] = tab2.et2*tau/2 + tab2.etcum*tab2.F2
# dmod2 = smf.ols("mydiff ~ 0 + C(ddi):F2",data=tab2,missing='drop').fit()
# epredN = np.sqrt(np.clip(dmod2.predict(tab2)*2/tau - tab2.etcum*tab2.F2*2/tau,0,np.inf))
# #%%
# r2_skipna(epredM/tab2.et_init,tab2.ET/tab2.et_init)
#%%

# term1 = -1/tau*tab2.F*tab2.G
# #c2 = (np.nanmean(jtab.ET) - np.nanmean(term1))/np.nanmean(jtab.F)
# c2 = tab2.et_init/tab2.f_init
# # sm0 = 10
# # c1 = np.sqrt(sm0*4)
# # c2 = 0.5*c1*np.sqrt(2/tau)
# epred20 = np.clip(term1 + c2*tab2.F,0,np.inf)
from statsmodels.stats.anova import anova_lm

biome_diff = anova_lm(smf.ols("tau_ddreg ~ C(combined_biome)",data=df_meta).fit())


#%%
# tab1dd["meandiff"] = (tab1dd.et2 + 2/tau*tab1dd.etcum*tab1dd.F2)*tau/2 / (tab1first.F2)
# #tab1dd["meandiff"] = (tab1dd.etnorm + 2/tau*tab1dd.etcum)*tau/2
# tab2 = pd.merge(tab1,tab1dd[["ddi","meandiff"]],how="left",on="ddi")
# tab2["etpred"] = np.sqrt(2/tau*tab2.F2*np.clip(tab2.meandiff-tab2.etcum,0,np.inf))
# tab1first["et_init"] = 1*tab1first.ET
# tab2 = pd.merge(tab2,tab1first[["ddi","et_init"]],how="left",on="ddi")
#%%
doLAInew = 0
if doLAInew:
    laimms = pd.read_csv("fluxnet_modisLAI_1km_sq.csv")
    laimms["datecode"] = laimms["system:index"].str.slice(stop=10)
    laimms["date"] = pd.to_datetime(laimms["datecode"],format="%Y_%m_%d").dt.date
    laihow = laimms.loc[laimms.SITE_ID==site_id].copy();
    laihow["daydiff"] = (pd.to_datetime(laihow.date) - np.datetime64("2000-01-01"))/np.timedelta64(1,"D")
    drange = np.arange(np.min(laihow.daydiff),np.max(laihow.daydiff))
    medsmooth = 0*drange
    for i in range(15,len(medsmooth)-15):
        medsmooth[i] = np.max(laihow["mean"][(laihow.daydiff >= drange[i-15])*(laihow.daydiff < drange[i+15])])
    medsmooth[-15:] = medsmooth[-16]
    medsmooth[:15] = medsmooth[15]
    dfull["daydiff"] = (pd.to_datetime(dfull.date) - np.datetime64("2000-01-01"))/np.timedelta64(1,"D")
    newdf = pd.DataFrame({"daydiff":drange,"newLAI":medsmooth/10})
    dfull2 = pd.merge(dfull,newdf,how="left",on="daydiff")
#%%
# laidiff = []
# for site_id in pd.unique(bigyear.SITE_ID):
#     dfull = bigyear.loc[bigyear.SITE_ID==site_id].copy()

#     for y0 in pd.unique(dfull.year_new):
#         dfy = dfull.loc[dfull.year_new==y0].copy()
#         laidiff.append(np.diff(dfy.LAI))
# laid2 = np.concatenate(laidiff)
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 07:00:10 2023

@author: natan
"""


import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import datetime
import scipy.interpolate
import h5py

#%%

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%
#dfEVI = pd.read_csv("evi_allyear_allsite.csv")
#dfEVI = pd.read_csv("evi_yearly_pheno_2cyc.csv")
dfEVI = pd.read_csv("evi_all_buff2500.csv")

#%%
def itodate(x):
    return np.datetime64("1970-01-01") + np.array(x)*np.timedelta64(1,"D")
#%%
datecols = ['Dormancy_1', 'Greenup_1', 'Maturity_1', 'MidGreendown_1',
       'MidGreenup_1', 'Peak_1',
       'Senescence_1']
datecols2 = [x[:-1]+'2' for x in datecols]
numcols = ['EVI_Amplitude_1', 'EVI_Area_1',
       'EVI_Minimum_1']
numcols2 = [x[:-1]+'2' for x in numcols]

datecols_order = ['Greenup_1', 'MidGreenup_1',
                             'Maturity_1', 'Peak_1','Senescence_1' ,
                             'MidGreendown_1','Dormancy_1']
datecols2_order = [x[:-1]+'2' for x in datecols_order]

#%%
dfEVI["row_index"] = range(len(dfEVI))
evi_long = pd.melt(dfEVI,id_vars=["SITE_ID","row_index"]+numcols,value_vars=datecols)
evi_long[numcols] /= 10000
#%%
ytemplate = np.array([0.15,0.5,0.9,1,0.9,0.5,0.15])
sidetemplate = np.array([-1,-1,-1,0,1,1,1])

evi_long["y_relative"] = evi_long.variable.replace(dict(zip(datecols_order,ytemplate)))
evi_long["side_of_peak"] = evi_long.variable.replace(dict(zip(datecols_order,sidetemplate)))
evi_long["y_actual"] = evi_long.y_relative*evi_long.EVI_Amplitude_1 + evi_long.EVI_Minimum_1
#%%
evi_long["dateEVI"] = itodate(evi_long.value)
#%%
evi_long2 = pd.melt(dfEVI,id_vars=["SITE_ID","row_index"]+numcols2,value_vars=datecols2)
evi_long2[numcols2] /= 10000
#%%
evi_long2["y_relative"] = evi_long2.variable.replace(dict(zip(datecols2_order,ytemplate)))
evi_long2["side_of_peak"] = evi_long2.variable.replace(dict(zip(datecols2_order,sidetemplate)))
evi_long2["y_actual"] = evi_long2.y_relative*evi_long2.EVI_Amplitude_2 + evi_long2.EVI_Minimum_2
#%%
evi_long2 = evi_long2.loc[np.isfinite(evi_long2.y_actual)].copy()
#%%
evi_long2["dateEVI"] = itodate(evi_long2.value)
#%%
evi_long = pd.concat([evi_long,evi_long2])
#%%
flux_data = pd.read_csv("gs_50_yearlyGS_mar8.csv",parse_dates =["date"])
#flux_data = pd.read_csv("gs_50_50_mar5.csv",parse_dates =["date"])

#%%
flux_data["day1970"] = (flux_data.date - datetime.datetime(1970,1,1)).astype('timedelta64[D]')
# #%%
# df2 = pd.merge(flux_data,gstab,on=["SITE_ID","year"],how="left")
# df2 = df2.loc[np.isfinite(df2.Peak_1)].copy()
# #%%
flux_data["EVIint"] = np.nan
#%%

# bif_data = pd.read_csv("fn2015_bif_tab_h.csv")
# #bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])].copy()
# bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])].copy()
# metadata = pd.read_csv("fluxnet_site_info_all.csv")

# useLAI = 0
# if useLAI:
#     lai_map = h5py.File("LAI_mean_monthly_1981-2015.nc4","r")
#     site_ilat = np.floor((np.array(bif_forest.LOCATION_LAT)+90)*4).astype(int)
#     site_ilon = np.floor((np.array(bif_forest.LOCATION_LONG)+180)*4).astype(int)
    
#     lai_allmo = np.array(lai_map["LAI"])[:,site_ilat,site_ilon]
#     lai_allmo[lai_allmo < 0] = np.nan
#     month_template = [8,9,10,11,12,1,2,3,4,5,6,7]
#     lai_pivot = lai_allmo.T.reshape(-1,1)[:,0]
#     lai_piv_tab = pd.DataFrame({"SITE_ID":np.repeat(bif_forest.SITE_ID,12),
#                                 "LAIclim":lai_pivot,
#                                 "month":np.tile(month_template,len(bif_forest))})
    
#     lai_int = []
#     for site_id in pd.unique(df_in.SITE_ID):
#         #print(site_id)
#         dfi = df_in.loc[df_in.SITE_ID==site_id].copy()
#         doy_arr = np.array(dfi.doy_raw)
#         lai_site_tab = lai_piv_tab.loc[lai_piv_tab.SITE_ID==site_id].copy().sort_values("month")
#         lai_arr = np.interp(doy_arr,  (np.arange(12)+0)/12*365,   np.array(lai_site_tab.LAIclim))
#         lai_int.append(lai_arr/np.max(lai_arr))
#     df_in["LAIint_rel"] = np.concatenate(lai_int)
#%%
newdf = []
for site_id in pd.unique(flux_data.SITE_ID):
    print(site_id)
    sitesel = flux_data.SITE_ID==site_id
    dfsub = flux_data.loc[sitesel].copy()
    evisub = evi_long.loc[evi_long.SITE_ID==site_id].sort_values("value").reset_index()
    if len(evisub)==0:
        continue
    myinterp = np.interp(np.array(dfsub.day1970),
                         np.array(evisub.value),np.array(evisub.y_actual),left=np.nan,right=np.nan)
    dfsub["EVIint"] = myinterp
    myinterp2 = np.interp(np.array(dfsub.day1970),
                         np.array(evisub.value),np.array(evisub.side_of_peak),left=np.nan,right=np.nan)
    dfsub["EVIside"] = myinterp2
    
    laifile = "lai_csv/lai_csv/" + "_".join(site_id.split("-")) + "_LAI_FLX15.csv"
    try:
        laidf = pd.read_csv(laifile,parse_dates=["Time"])
    except:
        continue
    dfsub = pd.merge(dfsub,laidf,left_on = "date",right_on="Time",how='left')
    
    newdf.append(dfsub)
newdf = pd.concat(newdf)
#%%
newdf.to_csv("gs50_varGS_evi_lai.csv")
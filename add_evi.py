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
flux_data = pd.read_csv("gs_50_laiGS_mar13.csv",parse_dates =["date"])
#flux_data = pd.read_csv("gs_50_50_mar5.csv",parse_dates =["date"])

#%%
flux_data["day1970"] = (flux_data.date - datetime.datetime(1970,1,1)).astype('timedelta64[D]')
# #%%
# df2 = pd.merge(flux_data,gstab,on=["SITE_ID","year"],how="left")
# df2 = df2.loc[np.isfinite(df2.Peak_1)].copy()
# #%%
flux_data["EVIint"] = np.nan
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
    newdf.append(dfsub)
newdf = pd.concat(newdf)
#%%
newdf.to_csv("gs50_mar13_lai_evi.csv")
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:02:51 2022

@author: nholtzma
"""

import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
#import pandas as pd
#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams['lines.linewidth']=3

plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%
datalen = 120

prob_rain = 0.05
mean_rain = 3

precip = (np.random.rand(datalen) < prob_rain).astype(float)
precip *= mean_rain/prob_rain
#%%
vpd = 0.75/100
k = 0.1
sw = -100+500
tau = 50
tau_seconds = tau*(60*60*24)

et_arr = np.zeros(datalen)
g_arr = np.zeros(datalen)
sm_arr = np.zeros(datalen)


sm = 0+500
for i in range(datalen):
    sm_arr[i] = sm
    gi = np.sqrt(2/tau_seconds*k/vpd*np.clip(sm-sw,0,np.inf)*1000/18) #cond in mol/m^2/s
    eti = gi*vpd * 18/1000*60*60*24
    g_arr[i] = gi
    et_arr[i] = eti
    sm += 0*precip[i] - eti

tau = 20
tau_seconds = tau*(60*60*24)

et_arr2 = np.zeros(datalen)
sm_arr2 = np.zeros(datalen)
g_arr2 = np.zeros(datalen)


sm = 0+500
for i in range(datalen):
    sm_arr2[i] = sm
    gi = np.sqrt(2/tau_seconds*k/vpd*np.clip(sm-sw,0,np.inf)*1000/18) #cond in mol/m^2/s
    eti = gi*vpd * 18/1000*60*60*24
    g_arr2[i] = gi

    et_arr2[i] = eti
    sm += 0*precip[i] - eti
#%%
gpp1 = 1 - np.exp(-g_arr/k)
gpp2 = 1 - np.exp(-g_arr2/k)
#%%

#%%
plt.figure(figsize=(14,5))

plt.subplot(1,3,1)
plt.plot(et_arr,label=r"$\tau=50$ days")
plt.plot(et_arr2,label=r"$\tau=20$ days")
plt.xlim(0,100)
plt.legend()

plt.title("ET (mm/day)")
plt.subplot(1,3,2)
plt.plot(gpp1)
plt.plot(gpp2)
plt.xlim(0,100)
plt.title("$A/A_{max}$")
plt.xlabel("Time (days)")

plt.subplot(1,3,3)
plt.plot(sm_arr/500-0.6,label=r"$\tau=50$ days")
plt.plot(sm_arr2/500-0.6,label=r"$\tau=20$ days")
plt.xlim(0,100)
plt.title("Soil moisture")

plt.tight_layout()

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/theory_fig2.svg",bbox_inches="tight")
#%%
amax = 10
g63 = 0.05
g_range = np.linspace(0,0.3,500)
a_range = amax*(1-np.exp(-g_range/g63))
plt.figure(figsize=(7,7))
plt.plot(g_range,a_range,color="green",linewidth=6,alpha=0.75)
plt.xlabel("g $(mol/m^2/s)$",fontsize=24)
plt.ylabel("A $(\mu mol/m^2/s)$",fontsize=24)
plt.axhline(10,color="blue",linestyle="--",linewidth=6)
plt.plot([0,0.04],[0,0.04*amax/g63],":",color="brown",linewidth=6)
plt.xlim(0,0.29)
plt.ylim(0,11)
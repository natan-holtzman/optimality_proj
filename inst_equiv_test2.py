# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:42:08 2023

@author: natan
"""


import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import statsmodels.api as sm
import scipy.optimize
#import glob
#import statsmodels.formula.api as smf

#import matplotlib as mpl
#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=24
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%
g_range = np.linspace(0.01,1.5,200).reshape(-1,1,1)
g_inc = g_range[1,0,0]-g_range[0,0,0]

vmax = 15
slope_at_0 = 110
a_exp = vmax*(1-np.exp(-g_range/vmax*slope_at_0))
a_prime = slope_at_0*np.exp(-g_range/vmax*slope_at_0)
a_pp = -slope_at_0**2/vmax*np.exp(-g_range/vmax*slope_at_0)

#%%
vpd_range = np.linspace(0.1,3,30).reshape(1,-1,1)
tau_day = 30
tau_s = tau_day*(60*60*24)
zsoil_mol = 1000*1000/18
sa_range = np.linspace(0.005,0.3,60).reshape(1,1,-1)
g_opt = np.sqrt(2/tau_s*vmax/slope_at_0*zsoil_mol*sa_range/(vpd_range/100))
#%%
# a = 1/tau_s
# B = vpd_range/100/zsoil_mol
# C = a/B
#%%
a_opt = vmax*(1-np.exp(-g_opt/vmax*slope_at_0))
ap_opt = slope_at_0*np.exp(-g_opt/vmax*slope_at_0)
app_opt = -slope_at_0**2/vmax*np.exp(-g_opt/vmax*slope_at_0)
#%%
#b = 1
#myratio = ap_opt/a_opt
#a = myratio/(b*g_opt**(b-1) + myratio*g_opt**b)
#Theta_guess = a_exp*(a*g_range**b)

f_fac = 1/(a_opt/ap_opt + g_opt)
f_power = np.exp(-1.6)*g_opt**-1.3
f_exp = np.exp(2 - 7.5*g_opt)
#f_fac = f_power

Theta_guess = a_exp*(f_fac*g_range)
whole_fun = a_exp - Theta_guess
fun_max = np.argmax(whole_fun,0) * g_inc + g_range[0]
#%%
#%%
plt.figure()
plt.plot(fun_max.reshape(-1,1),g_opt[0,:,:].reshape(-1,1),'.')
plt.plot([0,1.5],[0,1.5])
#%%
plt.figure()
plt.plot(g_range[:,0,0],Theta_guess[:,15,30])
plt.xlabel("g (mol $H_2O$/$m^2$/s)")
plt.ylabel("Cost function ($\mu mol$ C/$m^2$/s)")
plt.title("$\Theta$(g) with environment fixed")
#%%
plt.figure()
plt.plot(vpd_range[0,:,0],Theta_guess[50,:,30],'o-')
plt.xlabel("VPD (kPa)")
plt.ylabel("Cost function ($\mu mol$ C/$m^2$/s)")
plt.title("$\Theta$(VPD) with other variables fixed")
#%%
plt.figure()
plt.plot(sa_range[0,0,:],Theta_guess[50,15,:],'o-')
plt.xlabel("s - $s_w$")
plt.ylabel("Cost function ($\mu mol$ C/$m^2$/s)")
plt.title("$\Theta$(soil moisture) with other variables fixed")



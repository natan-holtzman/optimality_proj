# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:42:08 2023

@author: natan
"""


import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
#import pandas as pd
#import statsmodels.api as sm
#import scipy.optimize
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
g_range = np.linspace(0.01,1.5,100).reshape(-1,1,1,1)
#g_inc = g_range[1,0,0]-g_range[0,0,0]

#e_range = np.linspace(1e-5,0.015,100).reshape(-1,1,1,1)
vpd_range = np.linspace(0.1,3,30).reshape(1,-1,1,1)

#g_range = e_range/(vpd_range/100)

vmax = 15
slope_at_0 = np.linspace(60,140,20).reshape(1,1,1,-1)
a_exp = vmax*(1-np.exp(-g_range/vmax*slope_at_0))
a_prime = slope_at_0*np.exp(-g_range/vmax*slope_at_0)
a_pp = -slope_at_0**2/vmax*np.exp(-g_range/vmax*slope_at_0)

#%%
tau_day = 30
tau_s = tau_day*(60*60*24)
zsoil_mol = 1000*1000/18
sa_range = np.linspace(0.005,0.3,40).reshape(1,1,-1,1)
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
n = 3

f_fac = 1/(a_opt/ap_opt*n*g_opt**(n-1) + g_opt**n)
#f_power = np.exp(-1.6)*g_opt**-1.3
#f_exp = np.exp(2 - 7.5*g_opt)
#f_fac = f_power

Theta_guess = a_exp*(f_fac*g_range**n)
whole_fun = a_exp - Theta_guess
#fun_max = np.argmax(whole_fun,0)
#%%
#%%
# plt.figure()
# plt.plot(fun_max.reshape(-1,1),g_opt[0,:,:].reshape(-1,1),'.')
# plt.plot([0,1.5],[0,1.5])
#%%
#dTdE = dTdg/dEdg
#dTdE = a_exp*f_fac/(vpd_range/100)
#evap = g_range*(vpd_range/100)
#%%
dTdg = f_fac*(a_exp*g_range**(n-1)*n + g_range**n*a_prime)
#dTdE = dTdg/(vpd_range/100)
#dAdE = a_prime/(vpd_range/100)*np.ones(Theta_guess.shape)
#%%
#mol h2o to mmol h2o
#e_range *= 1000
#dAdE /= 1000
#dTdE /= 1000

#%%
fig,ax_all = plt.subplots(1,3,figsize=(18,5))

ax = ax_all[0]
ax.plot(g_range[:,0,0,0] , dTdg[:,15,15,10],"r-.",label="Marginal penalty $d\Theta/dg$")#,label="Low VPD")
ax.plot(g_range[:,0,0,0] , dTdg[:,25,15,10],"r")#,label="High VPD")
ax.plot(g_range[:,0,0,0] , a_prime[:,0,0,10],"b-.",label="Marginal gain $dA/dg$")
ax.plot(g_range[:,0,0,0] , a_prime[:,0,0,10],"b")

# ax.plot(e_range[:,0,0,0] , dTdE[:,0,15,10],"r-.",label="Marginal penalty $d\Theta/dE$")#,label="Low VPD")
# ax.plot(e_range[:,0,0,0] , dTdE[:,1,15,10],"r")#,label="High VPD")
# ax.plot(e_range[:,0,0,0] , dAdE[:,0,15,10],"b-.",label="Marginal gain $dA/dE$")
# ax.plot(e_range[:,0,0,0] , dAdE[:,1,15,10],"b")

ax.set_ylim(0,100)
ax.set_xlim(0,0.5)


#ax.plot()
#fig.legend()
ax.set_xlabel("g")
ax.set_ylabel("Marginal gain and penalty")
ax.set_title("Increasing VPD")

fig.legend(loc="upper center",bbox_to_anchor=(0.5,-0.02),ncols=2)


ax = ax_all[1]

ax.plot(g_range[:,0,0,0] , dTdg[:,20,15,5],"r-.")#,label="Low VPD")
ax.plot(g_range[:,0,0,0] , dTdg[:,20,15,15],"r")#,label="High VPD")
ax.plot(g_range[:,0,0,0] , a_prime[:,0,0,5],"b-.")
ax.plot(g_range[:,0,0,0] , a_prime[:,0,0,15],"b")

ax.set_ylim(0,100)
ax.set_xlim(0,0.5)



#ax.plot()
#ax.legend()
ax.set_xlabel("g")
#ax.ylabel("Marginal gain and penalty")
ax.set_title("Increasing $C_a$")

ax = ax_all[2]

ax.plot(g_range[:,0,0,0] , dTdg[:,20,20,10],"r-.")#,label="Low VPD")
ax.plot(g_range[:,0,0,0] , dTdg[:,20,15,10],"r")#,label="High VPD")
ax.plot(g_range[:,0,0,0] , a_prime[:,0,0,10],"b-.")
ax.plot(g_range[:,0,0,0] , a_prime[:,0,0,10],"b")

ax.set_ylim(0,100)
ax.set_xlim(0,0.5)



#ax.plot()
#ax.legend()
ax.set_xlabel("g")
#ax.ylabel("Marginal gain and penalty")
ax.set_title("Increasing soil drought")
#%%



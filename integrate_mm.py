# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:09:01 2023

@author: natan
"""

import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import statsmodels.api as sm
import scipy.optimize
import scipy.special
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
g_range = np.linspace(0.0,0.3,500)#.reshape(-1,1,1)
g_inc = g_range[1]-g_range[0]

# vmax = 15
# slope_at_0 = 110
# a_exp = vmax*(1-np.exp(-g_range/vmax*slope_at_0))
# a_prime = slope_at_0*np.exp(-g_range/vmax*slope_at_0)
# a_pp = -slope_at_0**2/vmax*np.exp(-g_range/vmax*slope_at_0)
V_m = 6.64
K_m = 0.0335
a_mm = V_m*g_range/(K_m+g_range)

#%%
def to_fit(pars):
    vmax, slope_at_0 = pars
    pred_exp = vmax*(1-np.exp(-g_range/vmax*slope_at_0))
    return pred_exp - a_mm
fit0 = np.array([V_m,V_m/K_m])
mymin = scipy.optimize.least_squares(to_fit,x0=fit0,method="lm",x_scale=np.abs(fit0))
#%%
vmax, slope_at_0 = mymin.x
fit_exp = vmax*(1-np.exp(-g_range/vmax*slope_at_0))
#%%
plt.figure()
plt.plot(g_range,fit_exp,label="Exponential",linewidth=3)
plt.plot(g_range,a_mm,label="Michaelis-Menten",linewidth=3)
plt.ylim(0,6.5)
plt.xlim(0,0.3)
plt.ylabel("A $(\mu mol$ $C/m^2/s)$")
plt.xlabel("$g^*$ $(mol$ $H_2 O/m^2/s)$")
plt.legend(loc="lower right")
#%%
vpd = 0.5
tau_day = 30
tau_s = tau_day*(60*60*24)
zsoil_mol = 500*1000/18 #mm to mol
sa_range = np.linspace(0,0.3,500)#.reshape(1,1,-1)
g_opt = np.sqrt(2/tau_s*vmax/slope_at_0*zsoil_mol*sa_range/(vpd/100))
#%%
B = vpd/100/zsoil_mol
gamma = 1/tau_s
a = gamma/(2*B)
#solve the ODE: dg/ds = a*(1 + K_m/x)
#%%
#inside_fac = zsoil_mol*sa_range/(vpd/100)/2/K_m/tau_s
inside_fac = a*sa_range/K_m

inside_w = np.exp(-1*inside_fac-1)
g_mm = -K_m *(1+np.real(scipy.special.lambertw(-inside_w,-1)))
#%%
xmod = np.stack((sa_range,np.sqrt(sa_range)),1)
mm_linmod = sm.OLS(g_mm,xmod,missing='drop').fit()
ls_pred = mm_linmod.predict(xmod)
#%%
#confirm numerically that the function is a solution
soil_inc = sa_range[1]-sa_range[0]
plt.figure()
plt.plot(g_mm,a*(1+K_m/g_mm))
plt.plot(g_mm[1:],np.diff(g_mm)/soil_inc)
plt.plot(ls_pred[1:], np.diff(ls_pred)/soil_inc)
#%%
plt.figure()
plt.plot(sa_range,g_opt,linewidth=3,label="Using exponential")
plt.plot(sa_range,g_mm,linewidth=3,label="Using Michaelis-Menten")
plt.plot(sa_range,ls_pred,
         "k:",linewidth=3,label="Linear+sqrt fit")
plt.xlabel("$s-s_w$ $(m^3/m^3)$")
plt.ylabel("$g^*$ $(mol$ $H_2 O/m^2/s)$")
plt.legend(loc="lower right")
#%%
fac2 = np.real(scipy.special.lambertw(-inside_w,-1))
#%%
ymod = np.stack((inside_fac,np.sqrt(inside_fac)),1)
ytopred = g_mm/K_m
#%%
mm2_linmod = sm.OLS(ytopred,ymod,missing='drop').fit()
mm2_logmod = sm.OLS(np.log(ytopred[1:]),sm.add_constant(np.log(inside_fac[1:])),missing='drop').fit()
#%%
plt.figure()
plt.plot(ytopred)
plt.plot(mm2_linmod.predict(ymod),":")
plt.plot(np.exp(mm2_logmod.predict(sm.add_constant(np.log(inside_fac)))),":")
#%%
plt.figure()
plt.plot(inside_fac)
plt.plot(g_mm/K_m - np.log(g_mm/K_m + 1),":")
plt.plot(g_opt/K_m - np.log(g_opt/K_m + 1),":")
#%%
#can fit to data using eqn
#let y = g/K_m - np.log(g/K_m + 1)
#then y = a*(s-s0)/K_m = Zsoil*(s-s0)/(2*tau_s*K_m*VPD/100)
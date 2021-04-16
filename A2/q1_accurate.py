# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:13:06 2020

@author: Alex Lee
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema

G = 6.67430e-11
Msun = 1988500e24 #kg
mearth = 5.97219e24 #kg

sec = 1
minu = 60*sec
hour = 60*minu
day = 24*hour
year = 365.242199*day
tyear = 10
T = tyear*year
nsteps0= 20*day
t = np.linspace(0.0, T, int(T/nsteps0))

r = np.array([0., 0., 0.])
v = np.array([0., 0., 0.])
a = np.array([0., 0., 0.])

au = 149597870700 #m
aupday = 149597870700/24/60/60 #m/s

xs = -5.871476350881860E-03*au
ys = 6.579099261042989E-03*au
zs = 8.190722904306934E-05*au ## z
rsun0 = np.array([xs, ys, ys])

vxs = -7.499359804658616E-06*aupday ## x
vys = -4.805753079522615E-06*aupday ## y
vzs = 2.213656602068544E-07*aupday ## z
vsun0 = np.array([vxs, vys, vzs])

xe = 9.886566130496588E-01*au
ye = -1.445653650721954E-01*au
ze = 8.438508460440852E-05*au
re0 = np.array([xe, ye, ze])

vxe = 2.302725583407780E-03*aupday
vye = 1.694428848894045E-02*aupday
vze = -1.042608823204765E-06*aupday
ve0 = np.array([vxe, vye, vze])

rcm = (Msun*rsun0 + mearth*re0)/(Msun+mearth)
vcm = (Msun*vsun0 + mearth*ve0)/(Msun+mearth)

rearth = re0 - rcm
vearth = ve0 - vcm
rsun = -mearth/Msun*rearth
vsun = -mearth/Msun*vearth

def TwoBodyEquations(w,t):
    r1=w[:3]
    r2=w[3:6]
    v1=w[6:9]
    v2=w[9:12]
    r=np.linalg.norm(r2-r1) #Calculate magnitude or norm of vector
    dv1bydt=G*mearth*(r2-r1)/r**3
    dv2bydt=G*Msun*(r1-r2)/r**3
    dr1bydt=v1
    dr2bydt=v2
    r_derivs=np.concatenate((dr1bydt,dr2bydt))
    derivs=np.concatenate((r_derivs,dv1bydt,dv2bydt))
    return derivs


init_params=np.array([rsun0,re0,vsun0,ve0]) #create array of initial params
init_params=init_params.flatten() #flatten array to make it 1D

sec = 1
minu = 60*sec
hour = 60*minu
day = 24*hour
year = 365.242199*day
tyear = 2000           ## which is reaching my computer's limit.
T = tyear*year
nsteps0= 1*day
t = np.linspace(0, T, int(T/nsteps0))

two_body_sol=odeint(TwoBodyEquations, init_params,t )

rsun = two_body_sol[:,:3]
rearth = two_body_sol[:,3:6]

vsun = two_body_sol[:, 6:9]
vearth = two_body_sol[:,9:12]

r_cm=(Msun*rsun + mearth*rearth)/(Msun+mearth)
v_cm=(Msun*vsun + vearth*ve0)/(Msun+mearth)

re_cm = rearth-r_cm
rs_cm = rsun-r_cm
ve_cm = vearth - v_cm
vs_cm = vsun - v_cm

days = t/day

re_mag = np.linalg.norm(re_cm, axis=1)
plt.plot(days, re_mag) 
plt.xlim(0, 365*10)

aphelion  = argrelextrema(re_mag, np.greater)
plt.scatter(days[aphelion], re_mag[aphelion], label = 'Aphelion')

perihelion = argrelextrema(-re_mag, np.greater)
plt.scatter(days[perihelion], re_mag[perihelion], label = 'Perihelion')
plt.xlabel('Day after 2020 Sept 14 ')
plt.ylabel('Distance to the sun [m] ')
plt.title('Distance change in 10 years example')
plt.legend()

plt.figure()
plt.plot(days[aphelion], re_mag[aphelion], label = 'Aphelion')
plt.plot(days[perihelion], re_mag[perihelion], label = 'Perihelion')
plt.xlabel('Days after 2020 Sept 14 ')
plt.ylabel('Distance to the sun [m] ')
plt.legend()
perihelion = np.array(perihelion)[0]
aphelion = np.array(aphelion)[0]
perihelion = perihelion[0: len(aphelion)]
eccen = (re_mag[aphelion] - re_mag[perihelion])/ (re_mag[aphelion] + re_mag[perihelion])
eyear = np.linspace(1, len(eccen), len(eccen))

plt.figure()
plt.title('Days between aphelions and aphelion')
years = np.linspace(1, len(aphelion)-1, len(aphelion)-1)
plt.plot(years,days[aphelion[1: len(aphelion)]]- days[aphelion[0: len(aphelion)-1]], label = 'Aphelion'  , marker= '*')
plt.plot(years,days[perihelion[1: len(perihelion)]]- days[perihelion[0: len(perihelion)-1]] , label = 'Perihelion', marker = '.')
plt.xlabel('Year after 2020')
plt.ylabel('Days of a year base on the aphelions difference')

plt.legend()
plt.figure()
plt.plot(eyear, eccen)
title = 'Eccentricity change in '+ str(tyear) + ' years'
plt.title(title)
plt.xlabel('Year after 2020')
plt.ylabel('Eccentricity ')


ve_mag = np.linalg.norm(ve_cm, axis=1)
vs_mag = np.linalg.norm(vs_cm, axis=1)


kinet = 0.5*mearth*ve_mag**2 + 0.5*Msun*vs_mag**2 
poten = G*mearth*Msun/np.linalg.norm(re_cm-rs_cm, axis=1)
energy = kinet - poten

re_mag = np.linalg.norm(re_cm, axis=1)
rs_mag = np.linalg.norm(rs_cm, axis=1)


L = mearth*np.cross(ve_cm, re_cm)+ Msun*np.cross(vs_cm, rs_cm)
L = np.linalg.norm(L, axis=1)
plt.figure()
plt.plot(days/365, abs(1-energy/energy[0])*100)
plt.plot(days/365, abs(1-L/L[0])*100)





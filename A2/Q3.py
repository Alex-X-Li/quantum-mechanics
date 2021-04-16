# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:54:31 2020

@author: Alex Lee
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.signal import argrelextrema

G = 6.67430e-11
Msun = 1988500e24 #kg
mearth = 5.97219e24 #kg
mjup = 1898.13e24


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

xj = 2.347699840428325E+00*au
yj = -4.555746878516208E+00*au
zj = -3.362641503393212E-02*au
rj0 = np.array([xj, yj, zj])

vxj = 6.615387200421259E-03*aupday
vyj = 3.815156141796302E-03*aupday
vzj = -1.637910576358818E-04*aupday
vj0 = np.array([vxj, vyj, vzj])


#Update CM 
r_cm0 = (Msun*rsun0 + mjup*rj0 + mearth*re0)/(Msun+mearth+mjup)
v_cm0 = (Msun*vsun0 + mjup*vj0 + mearth*ve0)/(Msun+mearth+mjup)


def ThreeBodyEquations(w,t):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    r12=np.linalg.norm(r2-r1)
    r13=np.linalg.norm(r3-r1)
    r23=np.linalg.norm(r3-r2)
    
    dv1bydt= G*mearth*(r2-r1)/r12**3 + G*mjup*(r3-r1)/r13**3
    dv2bydt= G*Msun*(r1-r2)/r12**3 + G*mjup*(r3-r2)/r23**3
    dv3bydt= G*Msun*(r1-r3)/r13**3 + G*mearth*(r2-r3)/r23**3
    
    dr1bydt= v1
    dr2bydt= v2
    dr3bydt= v3
    
    r12_derivs = np.concatenate((dr1bydt,dr2bydt))
    r_derivs = np.concatenate((r12_derivs,dr3bydt))
    v12_derivs = np.concatenate((dv1bydt,dv2bydt))
    v_derivs = np.concatenate((v12_derivs,dv3bydt))
    derivs = np.concatenate((r_derivs,v_derivs))
    return derivs

init_params=np.array([rsun0,re0,rj0,vsun0,ve0,vj0]) #Initial parameters
init_params=init_params.flatten()

sec = 1
minu = 60*sec
hour = 60*minu
day = 24*hour
year = 365.242199*day
tyear = 100           ## which is reaching my computer's limit.
T = tyear*year
nsteps0= 2*hour
t = np.linspace(0, T, int(T/nsteps0))

three_body_sol=odeint(ThreeBodyEquations, init_params,t )

rsun = three_body_sol[:,:3]
rearth = three_body_sol[:,3:6]
rjup = three_body_sol[:,6:9]

vsun = three_body_sol[:, 9:12]
vearth = three_body_sol[:,12:15]
vjup = three_body_sol[:,15:18]

r_cm=(Msun*rsun + mjup*rjup + mearth*rearth)/(Msun+mearth+mjup)
v_cm=(Msun*vsun + vjup*vj0 + vearth*ve0)/(Msun+mearth+mjup)

re_cm = rearth-r_cm
rs_cm = rsun-r_cm
rj_cm = rjup-r_cm

ve_cm = vearth - v_cm
vs_cm = vsun - v_cm
vj_cm = vjup - v_cm

re_mag = np.linalg.norm(re_cm, axis=1)
days = t/day
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
plt.scatter(years,days[aphelion[1: len(aphelion)]]- days[aphelion[0: len(aphelion)-1]], label = 'Aphelion'  , marker= '*')
plt.scatter(years,days[perihelion[1: len(perihelion)]]- days[perihelion[0: len(perihelion)-1]] , label = 'Perihelion', marker = '.')
plt.xlabel('Year after 2020')
plt.ylabel('Days of a year base on the aphelions difference')

plt.legend()
plt.figure()
plt.plot(eyear, eccen)
title = 'Eccentricity change in '+ str(tyear) + ' years'
plt.title(title)
plt.xlabel('Year after 2020')
plt.ylabel('Eccentricity ')

plt.figure()
plt.plot(re_cm[:, 0],re_cm[:, 1])
plt.plot(rs_cm[:, 0],rs_cm[:, 1], marker='o')
plt.plot(rj_cm[:, 0],rj_cm[:, 1])



# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 01:30:12 2020

@author: Alex Lee
"""

import scipy as sci
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema

G = 6.67430e-11
Msun = 1988500e24 #kg
mearth = 5.97219e24 #kg
au = 149597870700 #m
aupday = 149597870700/24/60/60 #m/s

sec = 1
minu = 60*sec
hour = 60*minu
day = 24*hour
year = 365.242199*day


xe = 9.886566130496588E-01*au
ye = -1.445653650721954E-01*au
ze = 8.438508460440852E-05*au
re0 = np.array([xe, ye, ze])

vxe = 2.302725583407780E-03*aupday
vye = 1.694428848894045E-02*aupday
vze = -1.042608823204765E-06*aupday
ve0 = np.array([vxe, vye, vze])

xs = -5.871476350881860E-03*au
ys = 6.579099261042989E-03*au
zs = 8.190722904306934E-05*au ## z
rsun0 = np.array([xs, ys, ys])

vxs = -7.499359804658616E-06*aupday ## x
vys = -4.805753079522615E-06*aupday ## y
vzs = 2.213656602068544E-07*aupday ## z
vsun0 = np.array([vxs, vys, vzs])


#Find Centre of Mass
r_com=(mearth*re0+Msun*rsun0)/(Msun+mearth)
re0 = re0 - r_com
rsun0 = rsun0 - r_com

v_com=(mearth*ve0+Msun*vsun0)/(Msun+mearth)
# ve0 = ve0 - v_com
# vsun0 = vsun0 - v_com


def TwoBodyEquations(w,t):
    r1=w[:3]
    r2=w[3:6]
    v1=w[6:9]
    v2=w[9:12]
    r=sci.linalg.norm(r2-r1) #Calculate magnitude or norm of r
    dv1bydt=G*Msun*(r2-r1)/r**3
    dv2bydt=G*mearth*(r1-r2)/r**3
    dr1bydt=v1
    dr2bydt=v2
    r_derivs = np.concatenate((dr1bydt,dr2bydt))
    derivs = np.concatenate((r_derivs,dv1bydt,dv2bydt))
    return derivs


init_params=np.array([re0, rsun0, ve0, vsun0]) #create array of initial params
init_params=init_params.flatten() #flatten array to make it 1D
'''
Define year and time step:

'''
tyear = 100
T = tyear*year
nsteps0= 2*hour
t = np.linspace(0.0, T, int(T/nsteps0))
#Run the ODE solver

two_body_sol=odeint(TwoBodyEquations, init_params, t)

rearth = two_body_sol[:,:3]
rsun = two_body_sol[:,3:6]
vearth = two_body_sol[:,6:9]
vsun = two_body_sol[:,9:12]

rcm=(mearth*rearth+Msun*rsun)/(Msun+mearth) ##center of mass
vcm=(mearth*vearth+Msun*vsun)/(Msun+mearth) ## center of mass

re_cm = rearth - rcm
ve_cm = vearth - vcm

rs_cm = -mearth/Msun*re_cm
vs_cm = -mearth/Msun*ve_cm

re_mag = np.sqrt(re_cm[:,0]**2+re_cm[:,1]**2+re_cm[:,2]**2)
rs_mag = np.sqrt(rs_cm[:,0]**2+rs_cm[:,1]**2+rs_cm[:,2]**2)

days = t/day

aphelion  = argrelextrema(re_mag, np.greater)
perihelion = argrelextrema(-re_mag, np.greater)
perihelion = np.array(perihelion)[0]
aphelion = np.array(aphelion)[0]
perihelion = perihelion[0: len(aphelion)]
eccen = (re_mag[aphelion] - re_mag[perihelion])/ (re_mag[aphelion] + re_mag[perihelion])
eyear = np.linspace(1, len(eccen), len(eccen))


res_mag = np.sqrt((re_cm[:,0] - rs_cm[:, 0])**2 + (re_cm[:,1] - rs_cm[:, 1])**2 + (re_cm[:,2] - rs_cm[:, 2])**2)
ve_mag = np.sqrt(ve_cm[:,0]**2+ve_cm[:,1]**2+ve_cm[:,2]**2)
vs_mag = np.sqrt(vs_cm[:,0]**2+vs_cm[:,1]**2+vs_cm[:,2]**2)

poten = -(G*Msun*mearth / (re_mag))
kine = 0.5*mearth*ve_mag**2 + 0.5*Msun*vs_mag**2
energy = poten + kine
L = mearth*ve_mag*re_mag+ Msun*vs_mag*rs_mag



sp = 1414  ### W/m^2
sa = 1322
S = 0.5* (sp*(re_mag[perihelion[0]]/re_mag)**2 +sa*(re_mag[aphelion[0]]/re_mag)**2)
plt.plot( days,S)
plt.xlim(0, 365*10)


plt.figure()
sigma = 5.670367e-8 ### Boltzmann constant
Tsun = 5772 ## K
radius_sun = 698000 *1000 ## m
phi = sigma*Tsun**4*radius_sun**2/re_mag**2
plt.plot(days,phi)
plt.xlim(0, 365*10)


n = 254 + days
theta = np.arcsin(0.39779*np.cos(np.radians(0.98565)*(n+10))+ np.radians(1.914)*np.sin(np.radians(0.98565)*(n-2)))
radius_earth = 6378.137*1000
phi_Equator = sigma*Tsun**4*radius_sun**2/(re_mag-radius_earth*(1-np.cos(theta)))**2
phi_NT = sigma*Tsun**4*radius_sun**2/(re_mag-radius_earth*(1-np.cos(theta+np.radians(23.26))))**2
phi_ST = sigma*Tsun**4*radius_sun**2/(re_mag-radius_earth*(1-np.cos(theta-np.radians(23.26))))**2
plt.figure()
# plt.plot(days, theta)
plt.xlim(0, 365*10)
# radius_earth = 6378.137
s_ratio = (abs(np.pi - theta)/np.pi ) - 2 *np.cos(theta)*np.sin(theta)/np.pi
phi_n = phi * s_ratio
plt.plot(days, phi_Equator-phi_NT, label = 'Northern Tropic')
plt.xlim(0, 365*5)
# plt.plot(days, phi_NT)

# plt.plot(days, phi_ST-phi_ST)
plt.xlim(0, 365*5)
plt.plot(days, phi_Equator-phi_ST, label = 'Southern Tropic' )
plt.legend()



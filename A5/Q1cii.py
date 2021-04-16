# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 03:20:16 2020

@author: Alex Lee
"""

import numpy as np
import matplotlib.pyplot as plt

m0 = 0.511e6    ## eV
c = 299792458  ##  m/s
hbar = 6.5821e-16   ##eV*s
ab = 5.29e-11      ##m
w = 1.5*ab
b = 0.5*ab


def f0(E1, v0, v1):
    k1_0 = np.sqrt(2*m0*(E1-v0)/(hbar**2*c**2)) 
    k2_0 = np.sqrt(2*m0*(v1-E1)/(hbar**2*c**2))
    f0 = np.cos(k1_0*w)*np.cosh(k2_0*b) + (k2_0**2 - k1_0**2)/(2*k2_0*k1_0)*np.sin(k1_0*w)*np.sinh(k2_0*b)
    return f0
    
    
def f1(E2, v0, v1):
    k1_1 = np.sqrt(2*m0*(E2-v0)/(hbar**2*c**2))
    k2_1 = np.sqrt(2*m0*(E2-v1)/(hbar**2*c**2))
    f1 = np.cos(k1_1*w)*np.cos(k2_1*b) - (k2_1**2 + k1_1**2)/(2*k2_1*k1_1)*np.sin(k1_1*w)*np.sin(k2_1*b)
    return f1

v1 = 360
v0 = 0
E = np.linspace(34, 40, 10000)
kl_low = np.arccos(f0(E, v0, v1))/np.pi

plt.plot(kl_low, E)
plt.plot(-kl_low, E)
plt.ylabel('Energy [eV]')
plt.xlabel(r'$kl/\pi$')
plt.grid()
plt.legend()

E0 = 37.31
t = 2.52

kl = np.linspace(-1, 1, 1000)
E_cos = E0 - 1*t*np.cos(kl*np.pi)
plt.plot(kl, E_cos)

# dEdt_pi1 = E[9783]-E[9781]/(kl_low[9783]-kl_low[9781])
# dEdt_pi2 = E[9781]-E[9779]/(kl_low[9781]-kl_low[9779])
# d2Edt2_pi = dEdt_pi1-dEdt_pi2/ (kl_low[9782]-kl_low[9780])
# print(d2Edt2_pi)

# dEdt_01 = E[1369]-E[1367]/(kl_low[1369]-kl_low[1367])
# dEdt_02 = E[1367]-E[1365]/(kl_low[1367]-kl_low[1365])
# d2Edt2_0 = dEdt_pi1-dEdt_pi2/ (kl_low[1368]-kl_low[1366])
# print(d2Edt2_0)

# dEdt_001 = E[1369]-E[1367]/(-kl_low[1369]+kl_low[1367])
# dEdt_002 = E[1367]-E[1365]/(-kl_low[1367]+kl_low[1365])
# d2Edt2_00 = dEdt_pi1-dEdt_pi2/ (+kl_low[1368]-kl_low[1366])
# print(d2Edt2_00)


from scipy.interpolate import UnivariateSpline


E = np.append(-np.sort(-E[1365:9783]), E[1365:9783])
kl_low = np.sort(np.append(-kl_low[1365: 9783], kl_low[1365:9783]))

plt.figure()
plt.plot(kl_low, E)
E_sp = UnivariateSpline(kl_low,E,s=0,k=3)
plt.plot(kl_low, E_sp(kl_low))
dE2_sp = E_sp.derivative(n = 2)
plt.figure()
plt.plot(kl_low, dE2_sp(kl_low))




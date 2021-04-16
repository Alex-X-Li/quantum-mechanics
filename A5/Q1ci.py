# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 02:31:21 2020

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

v0 = 0
v1 = 360
E1 = np.linspace(0.1, v1-0.1, 3600)
E2 = np.linspace(v1+0.1, 700, 3600)

f_0 = f0(E1, v0, v1)
f_1 = f1(E2, v0, v1)


plt.plot(E1, f_0, label = r'$E<V_1$')
plt.plot(E2, f_1, label = r'$E>V_1$')
plt.xlabel('Energy [eV]')
plt.grid()
plt.legend()

plt.figure()

E1 = np.linspace(0.1, v1-0.1, 360000)
E2 = np.linspace(v1+0.1, 1400, 360000)

kl1 = np.arccos(f0(E1, v0, v1))/np.pi
kl2 = np.arccos(f1(E2, v0, v1))/np.pi

plt.plot( kl1 ,E1, c = 'steelblue')
plt.plot( -kl1 ,E1, c = 'steelblue')
plt.plot( kl2,E2, c = 'steelblue',  label = r'$V_1 =360$')
plt.plot( -kl2,E2, c = 'steelblue')
plt.ylabel('Energy [eV]')
plt.xlabel(r'$kl/\pi$')


v1 = 0.00001
kl100 = np.arccos(f0(E1, v0, v1))/np.pi
kl200 = np.arccos(f1(E2, v0, v1))/np.pi

plt.plot( kl100 ,E1, c = 'red')
plt.plot( -kl100 ,E1, c = 'red')
plt.plot( kl200, E2, c = 'red',  label = r'$V_1 =0$')
plt.plot( -kl200, E2, c = 'red',  label = r'$V_1 =0$')
plt.grid()
plt.legend()
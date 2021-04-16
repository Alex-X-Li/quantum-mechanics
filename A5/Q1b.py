# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 23:31:56 2020

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

v1 = 360 ##eV
E1 = np.linspace(0.1, 359.99, 3600) ## 0 to 360 eV
E2 = np.linspace(360.01, 700, 3600) ## 360 to 700 eV


k1_0 = np.sqrt(2*m0*E1/(hbar**2*c**2)) 
k1_1 = np.sqrt(2*m0*E2/(hbar**2*c**2))

k2_0 = np.sqrt(2*m0*(v1-E1)/(hbar**2*c**2))
k2_1 = np.sqrt(2*m0*(E2-v1)/(hbar**2*c**2))

f0 = np.cos(k1_0*w)*np.cosh(k2_0*b) + (k2_0**2 - k1_0**2)/(2*k2_0*k1_0)*np.sin(k1_0*w)*np.sinh(k2_0*b)
f1 = np.cos(k1_1*w)*np.cos(k2_1*b) - (k2_1**2 + k1_1**2)/(2*k2_1*k1_1)*np.sin(k1_1*w)*np.sin(k2_1*b)


plt.plot(E1, f0, label = r'$E<V_1$')
plt.plot(E2, f1, label = r'$E>V_1$')
plt.xlabel('Energy [eV]')
plt.grid()
plt.legend()

E2 = np.linspace(360.01, 1400, 36000)
k1_0 = np.sqrt(2*m0*E1/(hbar**2*c**2)) 
k1_1 = np.sqrt(2*m0*E2/(hbar**2*c**2))

k2_0 = np.sqrt(2*m0*(v1-E1)/(hbar**2*c**2))
k2_1 = np.sqrt(2*m0*(E2-v1)/(hbar**2*c**2))
f1 = np.cos(k1_1*w)*np.cos(k2_1*b) - (k2_1**2 + k1_1**2)/(2*k2_1*k1_1)*np.sin(k1_1*w)*np.sin(k2_1*b)

z0 = np.arccos(f0)/np.pi
z1 = np.arccos(f1)/np.pi
plt.figure()
plt.scatter(z0, E1marker = '.')
plt.scatter(z1, E2,  marker = '.')

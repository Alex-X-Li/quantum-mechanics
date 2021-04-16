# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:18:21 2020

@author: Alex Lee
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def delta(m, n):
    if n == m:
        delta = 1
    else:
        delta = 0
    return delta
    

def j4(n):
    
    if n == 0:
        return 1/80
    else:
        return (1+(-1)**n)/2*(1/(n*np.pi)**2 - 24/(n*np.pi)**4)
        
    
def fij(i, j):
    return j4(i-j) - j4(i+j)

def isw(s, size):
    i = 0
    hij = np.zeros((size, size))
    while i < size:
        j = 0
        while j < size: 
            elem = (j+1)**2*delta(i+1, j+1) + np.pi**4*s**6*fij(i+1, j+1) 
            hij[i, j] = elem
            j = j+1
        i = i+1
           
    eigenValues, eigenVectors = la.eig(hij)
    eigenValues = np.real(eigenValues)
    
    index = np.linspace(0, size, size)
    
    idx = eigenValues.argsort()[::1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    return hij, eigenValues, eigenVectors, index

hij1, es1, evs1, ns1 = isw(1, 800)
plt.plot(ns1, es1, label = 's=1')

hij2, es2, evs2, ns2 = isw(2, 800)
plt.plot(ns2, es2, label = 's=2')

hij5, es5, evs5, ns5 = isw(5, 800)
plt.plot(ns5, es5, label = 's=5')

hij10, es10, evs10, ns10 = isw(10, 800)
plt.plot(ns10, es10, label = 's=10')
plt.xlim(0, 400)
plt.ylim(0, 8000)
plt.legend()

print(es1[0]/1, es2[0]/2**2, es5[0]/5**2, es10[0]/10**2)



# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:42:26 2020

@author: Alex Lee
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def eigenstate(size):
    rho = 50
    v = np.pi**2*rho**2/48
    
    
    delta = np.identity(size)
    
    n = 0
    m = 0
    matrix1 = np.zeros((size, size))
    
    while n < size:    
        while m < size:        
            elem = (m+1)**2+v*(1 - 6/(np.pi*(m+1))**2)        
            matrix1[n, m] = delta[n, m] * elem        
            m = m+1
        m = 0
        n = n + 1
        
    matrix2 = np.zeros((size, size))
    n = 0
    m = 0
    while n < size:    
        while m < size:
    
            if m != n:
                elem = 0.25*rho**2*((-1)**(m+n+2)+1)*(1/(n-m)**2-1/(n+m+2)**2)
            else:
                elem = 0
                   
            matrix2[n, m] = (1-delta[n, m]) * elem        
            m = m+1        
        m = 0
        n = n + 1        
                      
    matrix = matrix1 + matrix2    
    eigenValues, eigenVectors = la.eig(matrix)
    eigenValues = np.real(eigenValues)
    
    index = np.linspace(0, size, size)
    
    idx = eigenValues.argsort()[::1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return eigenValues, eigenVectors, index

e50, ev50, n50 = eigenstate(50)
plt.plot(n50, e50)

e100, ev100, n100 = eigenstate(100)
plt.plot(n100, e100)

e200, ev200, n200 = eigenstate(200)
plt.plot(n200, e200)


def numerical(eigenV, order):
    x = np.linspace(0, 1, 100)
    n = 1
    psi = 0
    while n < order:
        basis = np.sqrt(2)*np.sin(n*np.pi*x)
        ci = eigenV[n-1]        
        psi = psi+ basis*ci
        n = n+1    
    return x, psi




x_n, psi_10 = numerical(ev50[:, 0], 10)
x_n, psi_20 = numerical(ev50[:, 0], 20)
x_n, psi_50 = numerical(ev50[:, 0], 50)

x = x_n
psi = (np.pi/2*50)**0.25*np.exp(-np.pi**2/4*50*(x-0.5)**2)

plt.figure()
plt.scatter(x, psi, marker='^', c = 'red')
plt.plot(x_n, -psi_10 )
plt.plot(x_n, -psi_20 )
plt.plot(x_n, -psi_50 )
 

   
    
    





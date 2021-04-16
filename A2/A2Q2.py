# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:17:15 2020

@author: Alex Lee
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def eigenstate(size, v):
      
    n = 0
    matrix1 = np.zeros((size, size))
    
    while n < size:   
        m = 0
        while m < size:
            if m == n:                
                elem = (n+1)**2 + v/5 + 2*v*(2/5)*np.sinc(2*(n+1)*(2/5))                 
                matrix1[n, m] =  elem        

            
            else:
                elem1 = np.sinc(((n+1)-(m+1))*2/5)-np.sinc((n+1+m+1)*2/5)
                elem = -(2*v*2/5*(elem1)*(((-1)**(n+m+2)+1)/2))  
                matrix1[n, m] =  elem                
            m = m + 1
            
        n = n + 1
        
    eigenValues, eigenVectors = la.eig(matrix1)
    eigenValues = np.real(eigenValues)
    
    index = np.linspace(0, size, size)
    
    idx = eigenValues.argsort()[::1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return eigenValues, eigenVectors, index

def numerical(eigenV, order):
    x = np.linspace(0, 1, 1000)
    n = 1
    psi = 0
    while n < order+1:
        basis = np.sqrt(2)*np.sin(n*np.pi*x)
        ci = eigenV[n-1]        
        psi = psi+ basis*ci
        n = n+1    
    return x, psi

e0, ev0, index0 = eigenstate(500, 0)
e200, ev200, index200 = eigenstate(200, 2000000)

x0, psi0 = numerical(ev0[:, 0], 200)
x_0, psi_0 = numerical(ev200[:, 0], 200)
x_1, psi_1 = numerical(ev200[:, 1], 200)
x_2, psi_2 = numerical(ev200[:, 2], 200)
x_3, psi_3 = numerical(ev200[:, 3], 200)

# x_6, psi_6 = numerical(ev200[:, 8], 500)
# x_7, psi_7 = numerical(ev200[:, 9], 500)

# plt.plot(x_6, psi_6**2)
# plt.plot(x_7, psi_7**2)

# plt.plot(x0, psi0)
plt.plot(x_0, psi_0)
plt.plot(x_1, psi_1)
plt.grid()
# plt.figure()
# plt.plot(x_2, psi_2)
# plt.plot(x_3, psi_3)



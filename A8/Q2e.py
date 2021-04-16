# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:41:31 2020

@author: Alex Lee
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def delta(m, n):
    if n == m:
        return 1
    else:
        return 0


def qho(size):

    i = 0
    kij = np.zeros((size, size))

    ij1 = np.zeros((size, size))
    ij2_m = np.zeros((size, size))
    ij2_p = np.zeros((size, size))
    ij4_m = np.zeros((size, size))
    ij4_p = np.zeros((size, size))
    while i < size:
        j = 0
        while j < size: 
            kij[i, j] = -(3**(1/3))/2*(np.sqrt((j+2)*(j+1))*delta(i, j+2)+np.sqrt(j*(j-1))*delta(i, j-2) - (2*j+1)*delta(i,j))
            
            ij4_p[i, j] = np.sqrt((j+1)*(j+2)*(j+3)*(j+4))*delta(i, j+4)
            ij4_m[i, j] = np.sqrt((j)*(j-1)*(j-2)*(j-3))*delta(i, j-4)
            
            ij2_p[i, j] = (np.sqrt(j**2*(j+2)*(j+1))+np.sqrt((j+2)*(j+1))+np.sqrt((j+2)**3*(j+1)))*2*delta(i,j+2)
            ij2_m[i, j] = (np.sqrt( (j-2)**2*(j-1)*j ) + np.sqrt( (j-1)*j ) + np.sqrt( (j-1)*j**3) )*2*delta( i,j-2 )
            
            ij1[i, j] = (4*(j**2+j)+ 1 + (j+1)*(j+2) + j*(j-1) )*delta(i, j)
            

            j = j+1
        i = i+1        
    
    vij = (ij4_p + ij4_m + ij2_p + ij2_m + ij1)*3**(1/3)/12
    hij = vij + kij
    eigenValues, eigenVectors = la.eig(hij)
    eigenValues = np.real(eigenValues)
    
    index = np.linspace(0, size-1, size)
    
    idx = eigenValues.argsort()[::1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    return hij, eigenValues, eigenVectors, index

hij1, eigenValues1, eigenVectors1, index1 = qho(1)
hij2, eigenValues2, eigenVectors2, index2 = qho(2)
hij5, eigenValues5, eigenVectors5, index5 = qho(5)
hij10, eigenValues10, eigenVectors10, index10 = qho(10)
hij20, eigenValues20, eigenVectors20, index20 = qho(20)
hij50, eigenValues50, eigenVectors50, index50 = qho(50)
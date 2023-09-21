# -*- coding: utf-8 -*-
"""

Created on Tue Oct 29 15:12:53 2019

This Module computes the Fermi Energy for Nat electrons with the secant method to find the root of a function

@author: ahonet
"""

import numpy as np
#from numba import jit

#@jit(nopython=True)
def secant_method(a1,a2,tol,max_it,Nat,EigenVectors,EigenEnergies,Nel,kbT):
    q1=np.zeros(Nat)
    q2=np.zeros(Nat)
    qmid=np.zeros(Nat)
    max_it = 500
    it = 0

    while (it<max_it and np.abs(a2-a1)>tol):
        it +=1
        q1=np.zeros(Nat)
        q2=np.zeros(Nat)
        qmid=np.zeros(Nat)
        
        q1 = np.einsum( 'ij,j->i', np.square(EigenVectors), 1/(np.exp((EigenEnergies-a1)/kbT)+1), optimize=True ) 
        q2 = np.einsum( 'ij,j->i', np.square(EigenVectors), 1/(np.exp((EigenEnergies-a2)/kbT)+1),  optimize=True) 
        
        diff1_Nel = np.sum(q1) -  Nat - Nel
        diff2_Nel = np.sum(q2) -  Nat - Nel      
        
        a3 = a2 - ( (diff2_Nel * (a2 - a1)) / (diff2_Nel - diff1_Nel) )
        a2, a1 = a3*1, a2*1
        
        qmid = np.einsum( 'ij,j->i', np.square(EigenVectors), 1/(np.exp((EigenEnergies-a3)/kbT)+1), optimize=True ) 
        
        diffmid_Nel = np.sum(qmid) - Nat - Nel
        
        if (np.abs(diffmid_Nel)<tol):
            FermiE=a3*1
            break
        elif (np.abs(diff1_Nel)<tol):
            FermiE=a1 *1
            qmid=q1*1
            break
        elif (np.abs(diff2_Nel)<tol):
            FermiE=a2*1
            qmid=q2*1
            break
        
    FermiE = a3 *1   
    return FermiE, qmid

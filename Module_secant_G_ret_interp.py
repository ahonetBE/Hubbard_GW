# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:12:53 2019

This Module computes the Fermi Energy for Nat electrons with the secant method to find the root of a function

@author: ahonet
"""

import numpy as np
#from numba import jit

#@jit(nopython=True)
def secant_method(a1,a2,tol,max_it,Nat,G_ret,Nel,kbT,omega,FermiE):
    it = 0
    
    q1 = 0
    q2 = 0
    qmid = 0
    a3 = 0
    
    twopi_w = 2 * np.pi * omega
    domega = np.abs(omega[0]-omega[1]) 
    domega_vect = np.diff(omega)
    domega_vect = np.append(domega_vect, domega_vect[len(domega_vect)-1])
    dos = -2 * np.einsum('iik->k', G_ret.imag)
    int_dos = np.sum(dos*domega_vect)
    
    Nat = int_dos/2
    #Nat = Nat/2
    Nel *= 2*Nat/G_ret.shape[0]

    while (it<max_it and ( np.abs(a2-a1)>=tol or np.abs(a2-a1)==np.inf)):
        it +=1
        
        FDa1 = 1/ (np.exp((twopi_w-a1)/kbT)+1)
        FDa2 = 1/ (np.exp((twopi_w-a2)/kbT)+1)
        
        q1_old = np.sum(dos*FDa1) * domega
        q2_old = np.sum(dos*FDa2) * domega
        
        q1 = np.sum(dos*FDa1*domega_vect)
        q2 = np.sum(dos*FDa2*domega_vect) 
        
        diff1_Nel = np.sum(q1) -  Nat - Nel
        diff2_Nel = np.sum(q2) -  Nat - Nel
       
        a3 = a2 - ( (diff2_Nel * (a2 - a1)) / (diff2_Nel - diff1_Nel) )
        a2, a1 = a3*1, a2*1
    
        FDa3 = 1/ (np.exp((twopi_w-a3)/kbT)+1)
        qmid_old = np.sum(dos*FDa3) * domega
        qmid = np.sum(dos*FDa3*domega_vect) 
        
        diffmid_Nel = np.sum(qmid) - Nat - Nel
        if (np.abs(diffmid_Nel)<tol):
            FermiE = a3*1
            break
        elif (np.abs(diff1_Nel)<tol):
            FermiE = a1 *1
            qmid = q1*1
            break
        elif (np.abs(diff2_Nel)<tol):
            FermiE = a2*1
            qmid = q2*1
            break
        #added on 30/09/21 to avoid dividing by 0 in iterations ( / (diff2_Nel - diff1_Nel) )
        elif np.abs((diff2_Nel - diff1_Nel))<tol and np.abs(diff2_Nel) < 1e-8:   
            FermiE = a1 *1
            qmid = q1*1
            break
        elif np.abs((diff2_Nel - diff1_Nel))<tol and np.abs(diff2_Nel) > 1e-6:
            a2 =  np.random.random()*.2#*1e-8
            a1 = 0
        

    return FermiE, qmid, it

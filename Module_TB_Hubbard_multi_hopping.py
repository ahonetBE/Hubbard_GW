# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:12:53 2019

This module defines the Hubbard Hamiltonian up to third nearest neighbours (N.N.) for the hopping term

@author: ahonet
"""


import numpy as np
from numpy.linalg import eigh
#from numba import jit

#@jit
def Tight_Binding_Hubbard(Nat, x, y, z, NN_limit_dist, NN_limit_dist_2, NN_limit_dist_3, on_site_eps, on_site_dopant, hopping, hopping_2, hopping_3, delta_hopping, q, Uhu):
###############################################################################
#  Nat=number of atoms;  x,y,z= coordinates of atoms;   
#   NN_limit_dist = limit distance between two Nearest Neighbours
#   on_site_eps= on site interaction
#   hopping=hopping coefficient
###############################################################################

    ###############################################################################
    #Calculation of the inter-atomic distances            
    ###############################################################################
    inter_at_dist = np.zeros((Nat,Nat),dtype=float)
    NN_matrix = np.zeros((Nat,Nat),dtype=float)
    NN_matrix_vect_count = np.zeros(Nat,dtype=int)
    Ham_kin_1 = np.zeros((Nat,Nat),dtype=float)
    Ham_kin_2 = np.zeros((Nat,Nat),dtype=float)
    Ham_kin_3 = np.zeros((Nat,Nat),dtype=float)
    
    for ind1 in range(Nat):
        for ind2 in range(ind1):
            inter_at_dist[ind1,ind2] = np.sqrt( np.square(x[ind1]-x[ind2]) + np.square(y[ind1]-y[ind2]) + np.square(z[ind1]-z[ind2]) )
            inter_at_dist[ind2,ind1] = inter_at_dist[ind1,ind2]   
            ###############################################################################
            #N.N. matrix (within the precedent for loop)
            ###############################################################################
            if ((inter_at_dist[ind1,ind2] <= NN_limit_dist) & (inter_at_dist[ind1,ind2] !=0)):
                NN_matrix[ind1,ind2] = 1
                NN_matrix[ind2,ind1] = 1
                
                NN_matrix_vect_count[ind1] += 1
                NN_matrix_vect_count[ind2] += 1
                
                Ham_kin_1[ind1,ind2] = hopping
                Ham_kin_1[ind2,ind1] = hopping
                
            if ((inter_at_dist[ind1,ind2] >= NN_limit_dist) & (inter_at_dist[ind1,ind2] <= NN_limit_dist_2)):
                Ham_kin_2[ind1,ind2] = hopping_2
                Ham_kin_2[ind2,ind1] = hopping_2
                
            if ((inter_at_dist[ind1,ind2] >= NN_limit_dist_2) & (inter_at_dist[ind1,ind2] <= NN_limit_dist_3)):
                Ham_kin_3[ind1,ind2] = hopping_3
                Ham_kin_3[ind2,ind1] = hopping_3
     
    
    ###############################################################################
    #Hamiltonian matrix, eigen-vectors and eigen-energies
    ###############################################################################
    for ind1 in range(Nat):
        for ind2 in range(Nat):
            if NN_matrix_vect_count[ind1] == 2 or NN_matrix_vect_count[ind2] == 2:
                if Ham_kin_1[ind1, ind2 ] != 0:
                    Ham_kin_1[ind1, ind2 ] += delta_hopping
                    #Ham_kin_1[ind2, ind1 ] += delta_hopping
    
    
    
    Ham_kin = Ham_kin_1 + Ham_kin_2 + Ham_kin_3 
    
    H_mat = np.block([ [Ham_kin + np.diag(on_site_dopant), np.zeros((Nat, Nat))], [np.zeros((Nat, Nat)), Ham_kin + np.diag(on_site_dopant)] ]) + Uhu * np.block([ [ q[Nat:] * np.identity(Nat), np.zeros((Nat, Nat))], [np.zeros((Nat, Nat)), q[0:Nat] * np.identity(Nat)] ])
    
    EigenEnergies , EigenVectors = eigh(H_mat)
    
    return EigenEnergies, EigenVectors, H_mat
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:44:34 2020

This module runs a single iteration of the GW computation, with the Green's function and the self-energy as outputs

@author: ahonet
"""

import time
import numpy as np
from scipy import interpolate
from numpy.linalg import inv
import Module_secant_G_ret_interp
from scipy.integrate import simps

def one_iter_GW_sigma(Nel, Nat, omega, omega_homo, EigenEnergies, FermiE_sec, kbT, bp, bm, G_ret_0, G_ret, polar, Coulomb_mat, computation_name, n_chunks=1):
    
    
    hbar = 1
    
    theta_param = 0.001
    
    twopi_w = omega*(2*np.pi)
    twopi_w_homo = omega_homo * 2 * np.pi
    FDw = 1/ (np.exp((twopi_w-FermiE_sec)/kbT)+1)
    BEw = 1/ (np.exp((twopi_w-FermiE_sec)/kbT)-1)
    
    ft_time = np.fft.fftfreq(omega_homo.shape[-1], (bp-bm)/len(omega_homo))
    ift_omega = np.zeros(len(omega_homo))
    lomega_2 = int(len(omega_homo)/2)
    lomega = int(len(omega_homo))
    ift_omega[0:lomega_2] = omega_homo[lomega_2:lomega]
    ift_omega[lomega_2:lomega] = omega_homo[0:lomega_2]
    sort = np.argsort(ift_omega)
    
    mask = [np.where(omega_homo ==omega[i]) for i in range(len(omega)) ] 
    mask = np.reshape(np.array(mask), len(mask))

    Nat = len(EigenEnergies)
    
    def RPA_pol_GW(ind):
        epsilon = np.identity(len(EigenEnergies)) - np.matmul( Coulomb_mat, polar[:, :, ind])
        polar_RPA = np.matmul( polar[:, :, ind], inv(epsilon))
        return polar_RPA
               
    #Compute RPA susceptibility and retared screened potential (W) that will be used in the computation of the self-energy 
    polar_RPA = np.einsum('ijk, kjl -> ilk', polar, inv(np.transpose(
        np.identity(len(EigenEnergies))[:, :, np.newaxis, ] - np.einsum('ij, jlk -> ilk', Coulomb_mat, polar, optimize=True)
        , axes=(2,0,1))
        ), optimize=True)
    
    W_ret = Coulomb_mat[:, :, np.newaxis] + np.einsum( 'ijk, jlk -> ilk', np.einsum('ij, jlk -> ilk', Coulomb_mat, polar_RPA, optimize=True), Coulomb_mat[:, :, np.newaxis], optimize=True)
  
    #Computing self-energy (Sigma) in chunks
    def Sigma_chunk(range_x):
        
        W_lesser = BEw[np.newaxis,np.newaxis,:] * (2j* W_ret[range_x, :, :].imag)
        W_greater = (1 + BEw[np.newaxis,np.newaxis,:]) * (2j* W_ret[range_x, :, :].imag)
            
                        
        W_lesser_homo_func = interpolate.interp1d(omega, W_lesser, axis=2, fill_value="extrapolate")
        W_lesser_homo = W_lesser_homo_func(omega_homo)
        
        W_greater_homo_func = interpolate.interp1d(omega, W_greater, axis=2, fill_value="extrapolate")
        W_greater_homo = W_greater_homo_func(omega_homo)
        
        W_lesser_time = np.fft.fft(W_lesser_homo, axis=2, norm=None)
        W_greater_time = np.fft.fft(W_greater_homo, axis=2, norm=None)
        
        G_lesser = -FDw[np.newaxis, np.newaxis, :] * (2j) * G_ret[range_x, :, :].imag
        G_greater = (1-FDw[np.newaxis, np.newaxis, :]) * (2j) * G_ret[range_x, :, :].imag
        
        G_lesser_homo_func = interpolate.interp1d(omega, G_lesser, axis=2, fill_value="extrapolate")
        G_lesser_homo = G_lesser_homo_func(omega_homo)
        
        G_greater_homo_func = interpolate.interp1d(omega, G_greater, axis=2, fill_value="extrapolate")
        G_greater_homo = G_greater_homo_func(omega_homo)
        
        G_greater_time = np.fft.fft(G_greater_homo,axis=2,norm=None)
        G_lesser_time = np.fft.fft(G_lesser_homo,axis=2,norm=None)
        
        theta_param = 0.001
        
        Sigma_ret_time =  hbar * 1j  * (1 - (1/(np.exp(ft_time[np.newaxis, np.newaxis, :]/theta_param)+1) )) * ( G_greater_time * W_greater_time - G_lesser_time * W_lesser_time ) 
        
        Sigma_ret = np.fft.ifft(Sigma_ret_time, axis=2, norm=None)*(bp-bm)/len(omega_homo)       
        Sigma_ret = Sigma_ret[:, :, sort]
        Sigma_ret = Sigma_ret[:, :, mask]
        
        time_greater = np.where(ft_time>0)[0]
        time_lesser = np.where(ft_time<0)[0]
        time_lesser = np.flip(time_lesser)[0:len(time_lesser)-1]
        
        Sigma_lesser_time = 1j  * G_lesser_time * W_lesser_time
        Sigma_greater_time = 1j * G_greater_time * W_greater_time
        
        return Sigma_ret
                             
    ind_cut = [list(range(i*int(Nat/n_chunks), i*int(Nat/n_chunks) + int(Nat/n_chunks) )) for i in range(n_chunks)]
        
    Sigma_ret = map(Sigma_chunk, ind_cut)
    Sigma_ret = list(Sigma_ret)
    Sigma_ret = np.reshape(np.asarray(Sigma_ret, dtype=complex), (Nat, Nat, Sigma_ret[0].shape[2]))
    
    #Computing new Green's function from G_0 and the self-energy
    G_ret_new = inv( inv(np.transpose(G_ret_0, axes=(2,0,1))) - np.transpose(Sigma_ret, axes=(2,0,1))  ) 
    G_ret_new = np.transpose(G_ret_new, axes=(1,2,0))

    return  G_ret_new, Sigma_ret

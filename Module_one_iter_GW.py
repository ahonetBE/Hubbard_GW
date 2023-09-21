# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:44:34 2020

This module runs a single iteration of the GW computation, with the Green's function as output

@author: ahonet
"""

import numpy as np
from scipy import interpolate
from numpy.linalg import inv
import Module_secant_G_ret_interp
from functools import partial
import concurrent.futures
from scipy.integrate import simps


def xmul(a, b):
    out = np.empty_like(a)
    for j in range(a.shape[0]):
        out[j] = np.dot(a[j], b[j])
    return out

#Defining functions to compute RPA susceptibility, screened interaction, self-energy and Green's function
def Sigma_chunk(range_x, BEw, W_ret, G_ret, omega, omega_homo, FDw, ft_time, sort, mask, hbar, bp, bm):
        
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
        
    return Sigma_ret
        
def calc_G_new(range_ind_w, G_ret_0, Sigma_ret):
    
    if isinstance(range_ind_w, int):
        range_ind_w = [range_ind_w]
    
    G_ret_new_chunk = inv( inv(np.transpose(G_ret_0[:, :, range_ind_w], axes=(2,0,1))) - np.transpose(Sigma_ret[:, :, range_ind_w], axes=(2,0,1))  ) 
    G_ret_new_chunk = np.transpose(G_ret_new_chunk, axes=(1,2,0)) 
    return G_ret_new_chunk
  
def calc_W_ret(range_ind_w, Coulomb_mat, polar_RPA):  
    W_ret_chunk = Coulomb_mat[:, :, np.newaxis] + np.einsum( 'ijk, jlk -> ilk', np.einsum('ij, jlk -> ilk', Coulomb_mat, polar_RPA[:, :, range_ind_w], optimize=True), Coulomb_mat[:, :, np.newaxis], optimize=True)
    return W_ret_chunk
    
    
def calc_RPA_chunk(range_ind_w, polar, EigenEnergies, Coulomb_mat):
    polar_RPA_chunk = np.einsum('ijk, kjl -> ilk', polar[:, :, range_ind_w], inv(np.transpose(
        np.identity(len(EigenEnergies))[:, :, np.newaxis ] - np.einsum('ij, jlk -> ilk', Coulomb_mat, polar[:, :, range_ind_w], optimize=True)
        , axes=(2,0,1))
        ), optimize=True)
    return polar_RPA_chunk  
    
def full_G_new(EigenEnergies, omega, G_ret_0, Sigma_ret, ind_cut_omega):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        G_ret_new_test = list(executor.map(partial(calc_G_new, G_ret_0=G_ret_0, Sigma_ret=Sigma_ret), ind_cut_omega))
    
    G_ret_new_array = np.zeros((len(EigenEnergies), len(EigenEnergies), len(omega)),dtype=complex)
    for ind in range(len(ind_cut_omega)):
        G_ret_new_array[:, :, ind_cut_omega[ind]] = np.asarray(G_ret_new_test[ind])
    return G_ret_new_array



#One full iteration for the GW computation, using the above-defined functions for the computation of RPA susceptibility, screened potential, self-energy and Green's function
def one_iter_GW(Nel, Nat, omega, omega_homo, EigenEnergies, FermiE_sec, kbT, bp, bm, G_ret_0, G_ret, polar, Coulomb_mat, GW_iter, computation_name, n_chunks):
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
    
    n_chunks_omega = 16
    ind_cut_omega = [list(range(i*int(len(omega)/n_chunks_omega), i*int(len(omega)/n_chunks_omega) + int(len(omega)/n_chunks_omega) )) for i in range(n_chunks_omega)]
        
    polar_RPA = np.transpose( xmul( np.transpose(polar, axes=(2,0,1) ), inv(np.transpose(
        np.identity(len(EigenEnergies))[:, :, np.newaxis ] - np.tensordot( Coulomb_mat, polar, axes = ([1], [0])) 
        , axes=(2,0,1)))) , axes=(1, 2, 0))
    W_ret = Coulomb_mat[:, :, np.newaxis] + np.tensordot( Coulomb_mat, np.transpose( np.tensordot(polar_RPA, Coulomb_mat, axes = ([1],[0])), axes=(0, 2, 1)), axes = ([1], [0])) 
   
    del(polar_RPA)
    
    ind_cut = [list(range(i*int(Nat/n_chunks), i*int(Nat/n_chunks) + int(Nat/n_chunks) )) for i in range(n_chunks)]
      
    with concurrent.futures.ThreadPoolExecutor() as executor:
        Sigma_ret = list(executor.map(partial(Sigma_chunk, BEw=BEw, W_ret=W_ret, G_ret=G_ret, omega=omega, omega_homo=omega_homo, FDw=FDw, ft_time=ft_time, sort=sort, mask=mask, hbar=hbar, bp=bp, bm=bm), ind_cut, chunksize=30)) ### attention chunksize!

    Sigma_ret = list(Sigma_ret)
    Sigma_ret = np.reshape(np.asarray(Sigma_ret, dtype=complex), (Nat, Nat, Sigma_ret[0].shape[2]))

    del(W_ret)

    G_ret_new = inv( inv(np.transpose(G_ret_0, axes=(2,0,1))) - np.transpose(Sigma_ret, axes=(2,0,1))  ) 
    G_ret_new = np.transpose(G_ret_new, axes=(1,2,0))

    del(Sigma_ret)
    
    return  G_ret_new      

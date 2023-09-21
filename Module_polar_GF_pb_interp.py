# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:12:43 2019

This module computes the polarization from the Green's functions as inputs

@author: ahonet
"""

import numpy as np
from scipy import interpolate
import Module_secant_G_ret_interp
import time
import concurrent.futures


#@jit
def polar_GF(iteration_GW, Nel, Nat, omega, omega_homo, FermiE_sec, kbT, EigenVectors, EigenEnergies, ieta, bp, bm, computation_name, cond_shift, n_chunks=1, G_ret=None, with_G_ret=False):
    
    #We have to consider (2pi*omega) as the argument of the Green's functions because
    #   of the definition of the FFT implemented in numpy (2*pi in the complex exponential)
    #       that's also why we have to plot polar_RA in function of (2pi*omega)  
 
    hbar = 1
    
    theta_param = 0.001   
    twopi_w = omega*(2*np.pi)
    twopi_w_homo = omega_homo * 2 * np.pi
    FDw = 1/ (np.exp((twopi_w-FermiE_sec)/kbT)+1)
    
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
    
    G_ad = np.zeros((Nat,Nat,len(omega)),dtype=complex) 

    if with_G_ret == False:         
        G_ad = np.einsum( 'il,jl,kl->ijk', EigenVectors, np.conj(EigenVectors),1/(twopi_w[:,np.newaxis]-EigenEnergies[np.newaxis,:]-ieta) , optimize=True)
        G_ret = np.einsum( 'il,jl,kl->ijk', EigenVectors, np.conj(EigenVectors),1/(twopi_w[:,np.newaxis]-EigenEnergies[np.newaxis,:]+ieta), optimize=True )
        
    else:
        for indw in range(len(omega)):    
            G_ad[:,:,indw] = np.transpose( np.conj( G_ret[:,:,indw] ))
    
    
    a1 = 0.15
    a2 = 0.2
    tol = 10e-7
    max_it = 1000
    FermiE_save = FermiE_sec * 1
    count = 0
    FermiE_sec, charge, iter_done = Module_secant_G_ret_interp.secant_method(a1,a2,tol,max_it,Nat,G_ret,Nel,kbT,omega,FermiE_sec)

    while ((np.isinf(FermiE_sec) or np.isnan(FermiE_sec) or iter_done == max_it) == True):
        count += 1
        a1 = FermiE_save - np.random.random() * 0.2
        a2 = FermiE_save + np.random.random() * 0.2
        FermiE_sec, charge, iter_done = Module_secant_G_ret_interp.secant_method(a1,a2,tol,max_it,Nat,G_ret,Nel,kbT,omega,FermiE_sec)    
    
    indices_FermiE = np.where(np.abs(twopi_w-FermiE_sec) == np.amin(np.abs(twopi_w-FermiE_sec)))
    ecart_ind = int(len(omega)/2) - int(indices_FermiE[0][0])
    shift_ind = 0
    shift_ind = ecart_ind*1
   
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write( 'Number of frequency indices between Fermi levels: ' + '\t' + str(ecart_ind) + '\n')
        
    FermiE_sec_keep = FermiE_sec *1
    FermiE_sec = FermiE_save * 1
    

    #The "cond_shift" value is the threshold above which we align the Fermi levels at each iteration. If the Fermi levels are close enough, we do not align them.
    #The "len_polyfit" is the range on which we fit the tail of the Green's function to extrapolate it at the end of the frequency grid (after Fermi level shifting).
    def shift_G_ret(range_x):
        len_polyfit = 200    
        G_ret_homo_func_range = interpolate.interp1d(omega, G_ret[range_x, :, :], axis=2, fill_value="extrapolate")
        G_ret_homo = G_ret_homo_func_range(omega_homo)
        #cond_shift = 100
        
        if shift_ind >cond_shift :
            if range_x[0] == 0:
                with open('result-%i.txt' %(computation_name), 'a') as f:
                    f.write( 'Effective shift: ' + '\t' + str(shift_ind) + '\n')
            G_ret_homo[:, :, shift_ind:] = G_ret_homo[:, :, 0:len(omega_homo)-shift_ind]
        
            for ind1 in range(len(range_x)):
                for ind2 in range(Nat):
                    z = np.polyfit(omega_homo[shift_ind:shift_ind+len_polyfit], G_ret_homo[ind1, ind2, shift_ind:shift_ind+len_polyfit], 3)
                    p = np.poly1d(z)
                    G_ret_homo[ind1, ind2, 0:shift_ind] = p(omega_homo[0:shift_ind])
    
        elif shift_ind < -cond_shift :
            abs_shift = - shift_ind
            if range_x[0] == 0:
                with open('result-%i.txt' %(computation_name), 'a') as f:
                    f.write( 'Effective shift:' + '\t' + str(abs_shift) + '\n')
                
            G_ret_homo[:, :, 0:len(omega_homo)-abs_shift] = G_ret_homo[:, :, abs_shift:]
        
            for ind1 in range(len(range_x)):
                for ind2 in range(Nat):
                    z = np.polyfit(omega_homo[len(omega_homo)-abs_shift-len_polyfit:len(omega_homo)-abs_shift], G_ret_homo[ind1, ind2, len(omega_homo)-abs_shift-len_polyfit:len(omega_homo)-abs_shift], 3)
                    p = np.poly1d(z)
                    G_ret_homo[ind1, ind2, len(omega_homo)-abs_shift:] = p(omega_homo[len(omega_homo)-abs_shift:])
            
            
        G_ret_shift = G_ret_homo[:, :, mask]
        
        return G_ret_shift
        
    if iteration_GW>0:
    
        ind_cut = [list(range(i*int(Nat/n_chunks), i*int(Nat/n_chunks) + int(Nat/n_chunks) )) for i in range(n_chunks)]
        
        G_ret_shift = map(shift_G_ret, ind_cut)
        G_ret_shift = list(G_ret_shift)
        G_ret_shift = np.reshape(np.asarray(G_ret_shift, dtype=complex), (Nat, Nat, G_ret_shift[0].shape[2]))
        G_ret = G_ret_shift
    
    #Polarizability computation from Green's functions in chunks (range_x)
    def polar_fft(range_x):
        
        FDw = 1/ (np.exp((twopi_w-FermiE_sec)/kbT)+1)
        
        G_lesser_RA = G_ret[range_x, :, :].imag * 1 + 0 * 1j
        G_greater_RA = G_ret[range_x, :, :].imag * 1 + 0 * 1j
        
        G_lesser_RA *= -FDw[np.newaxis, np.newaxis, :] * (2j) 
        G_greater_RA *= (1-FDw[np.newaxis, np.newaxis, :]) * (2j) 
            
        
        G_lesser_homo_func = interpolate.interp1d(omega, G_lesser_RA, axis=2, fill_value="extrapolate")
        G_lesser_homo = G_lesser_homo_func(omega_homo)
        
        G_greater_homo_func = interpolate.interp1d(omega, G_greater_RA, axis=2, fill_value="extrapolate")
        G_greater_homo = G_greater_homo_func(omega_homo)

        G_greater_time_RA = np.fft.fft(G_greater_homo,axis=2,norm=None)
        G_lesser_time_RA = np.fft.fft(G_lesser_homo,axis=2,norm=None)
        
        polar_time_RA = G_greater_time_RA * 0

        ft_time_pos = [np.where(ft_time > 0) ] 
        ft_time_pos = np.reshape(np.array(ft_time_pos), len(ft_time_pos[0][0])) 
    
        polar_time_RA[:, :, ft_time_pos] =  - hbar * 1j * (1 - (1/(np.exp(ft_time[ft_time_pos]/theta_param)+1) )) * ( np.conj(G_greater_time_RA[:,:,ft_time_pos]) * G_lesser_time_RA[:,:,ft_time_pos] - np.conj(G_lesser_time_RA[:,:,ft_time_pos]) * G_greater_time_RA[:,:,ft_time_pos] )
            
        polar_RA = 2 * np.fft.ifft(polar_time_RA,axis=2,norm=None)*(bp-bm)/len(omega_homo)
        polar_RA = polar_RA[:, :, sort]/2
        polar_RA = polar_RA[:, :, mask]

        return polar_RA
    
    if iteration_GW>0:
        
        start_polar_paral = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            polar = list(executor.map(polar_fft, ind_cut)) ### attention chunksize!

        polar = list(polar)
        polar = np.reshape(np.asarray(polar, dtype=complex), (Nat, Nat, polar[0].shape[2]))
        end_polar_paral = time.time() 
            
    else:
        polar=0
   
    
    return FermiE_sec, FermiE_sec_keep, charge, G_ret, polar

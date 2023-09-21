# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:52:42 2020
This module is the core-module for GW computation.

Before the main function, we define several useful functions for different quantities

In the main function "calc_GW":
    - physical quantities for the MF approximation are computed from Green's functions, output as .npz files
    - in the while loop with condition on total energy or Green's functions, the GW iterations are performed using the module "module_one_iter_GW" at each iteration.
    - at different stages, if the iteration is 1, the results are set in .npz files as G0W0 results 
    - after the self-consistency is achieved (after the while loop), the GW quantities are computed and set as outputs in .npz files
@author: ahonet
"""

import numpy as np
from numpy.linalg import inv
import time
import Module_polar_GF_pb_interp
import Module_one_iter_GW
import Module_one_iter_GW_sigma
from scipy.integrate import simps


#Defining a multiplication operation (that better parallelizes)
def xmul(a, b):
    out = np.empty_like(a)
    for j in range(a.shape[0]):
        out[j] = np.dot(a[j], b[j])
    return out

#Function to compute total potential
def phi_tot_calc_hub(ind):
    epsilon = np.identity(len(EigenEnergies)) - xmul( One_over_r_Hartree, polar[:, :, ind])
    phi_tot = np.dot(inv(epsilon), phi_ext_mat_hub[:, ind])
    return phi_tot

#Main function in the GW module, implementing the GW self-consistency.
def calc_GW(x, y, z, Uhu, prop_u, on_site, Nel, bp, bm, Nat, omega, omega_homo, FermiE_sec, kbT, EigenVectors, EigenEnergies, ieta, H_mat, FermiE_before, calc_abs, computation_name, cond_shift, GW_toler, GW_max_iteration, n_chunks=1):
    
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('\n' +'\n' + '#################################################################################' + '\n')
        f.write('###############################################'  + '\n')
        f.write('Starting GW module' + '\n')
        f.write('###############################################' + '\n')
        f.write('#################################################################################'  + '\n' + '\n')
    
    #Useful constants and variables definitions
    light_vel = 3e8/(2.18e6)    
    x_s = np.concatenate((x, x), axis=None)     
    y_s = np.concatenate((y, y), axis=None) 
    z_s = np.concatenate((z, z), axis=None) 
    One_over_r_Hartree = 1 / ( (1.42/0.528) * np.sqrt(np.square(x[:,np.newaxis]-x[np.newaxis,:]) + np.square(y[:,np.newaxis]-y[np.newaxis,:]) + np.square(z[:,np.newaxis]-z[np.newaxis,:]) ) )
    np.fill_diagonal(One_over_r_Hartree, 0.58)    
    One_over_r_Hartree = np.block([
                                [One_over_r_Hartree, One_over_r_Hartree],
                                [One_over_r_Hartree, One_over_r_Hartree]
                                ])
    
    U_hub = Uhu
    U_mat = np.block([
                                [np.zeros((Nat, Nat)),   U_hub * np.identity(Nat) ],
                                [U_hub * np.identity(Nat), np.zeros((Nat, Nat)) ]
                                ])
    Coulomb_mat = 0 * One_over_r_Hartree + 1 * U_mat
    iteration = 0
    
    #First Green's function computation
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('\n' + '###############################################' + '\n')
        f.write('First Green''s function evaluation' + '\n')
        f.write('###############################################' + '\n')
        
    FermiE_tot = 0.
    FermiE_sec, FermiE_sec_keep, charge, G_ret, polar = Module_polar_GF_pb_interp.polar_GF(iteration, Nel, Nat, omega, omega_homo, FermiE_sec, kbT, EigenVectors, EigenEnergies, ieta, bp, bm, computation_name, cond_shift, n_chunks)
    
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write( '\n' + 'Number of electrons: ' + str(charge) + '\n')
    
    EigenEnergies -= FermiE_sec_keep
    FermiE_sec = 0.
    FermiE_tot += FermiE_sec_keep
    
    FDw = 1/ (np.exp((2*np.pi*omega-FermiE_sec)/kbT)+1)

    dens_mat_MF = -2 * simps(G_ret.imag*FDw, omega)
    H_mat_TB =  H_mat * 1
    np.fill_diagonal(H_mat_TB, 0.)
    G_lesser = -2j * G_ret.imag * FDw
    Trace_G = -2 * np.trace( G_ret.imag, axis1=0, axis2=1)
    E_gm_mu =  1/2 * simps(Trace_G * (omega + (FermiE_before+FermiE_tot)/(2*np.pi)) * FDw, omega) * 2 * np.pi
    second_term_H_TB = np.trace( np.matmul(H_mat_TB, dens_mat_MF)) /2    
    Energy_MF_GM = (E_gm_mu+second_term_H_TB)*27.21
    
    #Computing the mean densities and their difference in MF
    q_sec_up_MF_simps = np.diag(dens_mat_MF)[0 : Nat]
    q_sec_down_MF_simps = np.diag(dens_mat_MF)[Nat:]
    spin_MF_simps = (q_sec_up_MF_simps - q_sec_down_MF_simps)/2
    magnetization_MF = np.sum(np.abs(spin_MF_simps))
    spin_MF = np.sum(spin_MF_simps)
    
    
    iteration += 1
    
    #First shifting the Fermi level to align to 0
    FermiE_sec, FermiE_sec_keep, charge, G_ret, polar = Module_polar_GF_pb_interp.polar_GF(iteration, Nel, Nat, omega, omega_homo, FermiE_sec, kbT, EigenVectors, EigenEnergies, ieta, bp, bm, computation_name, cond_shift, n_chunks)
    
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('\n' + '###############################################' + '\n')
        f.write('After aligning Fermi level (MF)' + '\n')
        f.write('###############################################' + '\n')
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write( '\n' + 'Number of electrons: ' + str(charge) + '\n')
    
    EigenEnergies -= FermiE_sec_keep
    FermiE_sec = 0.
    FermiE_tot += FermiE_sec_keep
    
    
    #Definition of parameters for the self-consistency GW cycle
    #toler_G = 1e-6
    #max_iteration = 50
    alpha2 = .75 
    alphapr = 0.
    alpha_ini = alpha2 * 1        
    ecart_rel = 1
    iteration = 0
    E_tot_gw = 0
    E_tot_gw_old = 1
    

    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('Mixing parametersfor the Green''s functions: ' +  str(alpha2)  + ' and ' + str(alphapr) + '\n')
        
        
    #Start of the GW self-consistency cycle
    while ((np.abs(E_tot_gw-E_tot_gw_old)>GW_toler) and (iteration < GW_max_iteration)):  
    
        start_loop = time.time()  
        iteration += 1
        
        with open('result-%i.txt' %(computation_name), 'a') as f:
            f.write( '\n' + '###############################################' + '\n')
            f.write('Number of GW iteration: ' + str(iteration) + '\t' + 'ecart ' +  str(ecart_rel) + '\t' + 'ecart E tot:' + str((E_tot_gw-E_tot_gw_old)*27.21) + '\n')
            f.write( '###############################################' + '\n')
            f.write('Green''s function deviation: ' +  str(ecart_rel) + '\n')
            f.write('Total energy deviation: ' + str((E_tot_gw-E_tot_gw_old)*27.21) + '\n')
      
        if iteration == 1:
            G_ret_0 = G_ret * 1
            FermiE_0 = FermiE_tot*1
        else:
            start_else = time.time()
            G_ret = G_ret_combili * 1                        
            FermiE_sec, FermiE_sec_keep, charge, G_ret, polar = Module_polar_GF_pb_interp.polar_GF(iteration, Nel, Nat, omega, omega_homo, FermiE_sec, kbT, EigenVectors, EigenEnergies, ieta, bp, bm, computation_name, cond_shift, n_chunks, G_ret, 1)
            FermiE_tot += FermiE_sec_keep
            
            E_tot_gw_old = E_tot_gw*1
            E_kin_old = second_term_H_TB * 1
            E_GW_old = E_gm_mu*1
            end_else = time.time()
            
        G_ret_new = Module_one_iter_GW.one_iter_GW(Nel, Nat, omega, omega_homo, EigenEnergies, FermiE_sec, kbT, bp, bm, G_ret_0, G_ret, polar, Coulomb_mat, iteration, computation_name, n_chunks)
        end_one_iter = time.time()    
  
        FDw = 1/ (np.exp((2*np.pi*omega-(FermiE_sec + FermiE_sec_keep))/kbT)+1)
        
        dens_mat = -2 * simps(G_ret.imag*FDw, omega)
        H_mat_TB =  H_mat * 1
        np.fill_diagonal(H_mat_TB, 0.)
        # from eq. 19 of Joost 2020 (symmetry dilemma...)
        Trace_G = -2 * np.trace( G_ret.imag, axis1=0, axis2=1)
        second_term_H_TB = np.trace( np.matmul(H_mat_TB, dens_mat)) /2
        E_gm_mu = 1/2 * simps(Trace_G * (omega + (FermiE_before+FermiE_0+FermiE_sec_keep)/(2*np.pi)) * FDw, omega) * 2 * np.pi
        E_tot_gw = E_gm_mu + second_term_H_TB
 
        dos = -2 * np.einsum('iik->k', G_ret.imag)
        nb_el = simps(dos*FDw, omega)
        int_dos = simps(dos, omega)
        with open('result-%i.txt' %(computation_name), 'a') as f:
            f.write( 'Number of electrons: ' + str(nb_el) + '\n')    
            f.write( 'Integral of DOS: ' + str(int_dos) + '\n')    
            f.write( 'LDOS: ' + str(np.diag(dens_mat)) + '\n')  
            
        
        #If iteration is 1, we set all the quantities to G0W0 computations and use it as outputs
        if iteration == 1:
            G_ret_G0W0, Sigma_GW_G0W0 = Module_one_iter_GW_sigma.one_iter_GW_sigma(Nel, Nat, omega, omega_homo, EigenEnergies, FermiE_sec, kbT, bp, bm, G_ret_0, G_ret, polar, Coulomb_mat, computation_name, n_chunks)
            dens_mat_G0W0 = -2 * simps(G_ret_G0W0.imag*FDw, omega)
            dos_G0W0 = -2 * np.einsum('iik->k', G_ret_G0W0.imag)
            L_dos_G0W0 = -2 * np.einsum('iik->ik', G_ret_G0W0.imag)
            
            correlated_di = -1j * np.transpose( np.matmul( np.transpose( Sigma_GW_G0W0, axes=[2,0,1]), np.transpose( G_ret_G0W0, axes=[2,0,1]) ), axes=[1,2,0]) 
            correlated_di = simps(correlated_di * (1-np.heaviside(omega, 1)), omega*2*np.pi)
            correlated_di /= Uhu
            correlated_di /= 2*np.pi
            total_double_occ_G0W0 = 1/2 * np.concatenate( (np.diag(dens_mat_G0W0)[Nat:] * np.diag(dens_mat_MF)[0:Nat],   np.diag(dens_mat_MF)[Nat:] * np.diag(dens_mat_G0W0)[0:Nat]) )
            total_double_occ_G0W0 -= np.diag( correlated_di.real )           
            
            np.savez('G0W0@MF double occupation - computation %i' %(computation_name), total_double_occ_G0W0 )
            np.savez('G0W0@MF correlated part double occupation - computation %i' %(computation_name), np.diag( correlated_di.real ) )
            np.savez('G0W0@MF DOS - computation %i' %(computation_name), omega*2*np.pi*27.21, dos_G0W0 )
            np.savez('G0W0@MF L DOS - computation %i' %(computation_name), omega*2*np.pi*27.21, L_dos_G0W0 )
            ind_5 = int(np.where(np.abs(omega*2*np.pi*27.21-5) == np.amin(np.abs(omega*2*np.pi*27.21-5))) [0])
            ind_m5 = int(np.where(np.abs(omega*2*np.pi*27.21+5) == np.amin(np.abs(omega*2*np.pi*27.21+5))) [0])
            np.savez('G0W0@MF G_ret spin up - computation %i' %(computation_name), omega[ind_m5:ind_5]*2*np.pi*27.21, G_ret_G0W0[0:Nat, 0:Nat, ind_m5:ind_5] )
            np.savez('G0W0@MF G_ret spin down - computation %i' %(computation_name), omega[ind_m5:ind_5]*2*np.pi*27.21, G_ret_G0W0[Nat:, Nat:, ind_m5:ind_5] )
        else:
            G_ret_iter, Sigma_GW_iter = Module_one_iter_GW_sigma.one_iter_GW_sigma(Nel, Nat, omega, omega_homo, EigenEnergies, FermiE_sec, kbT, bp, bm, G_ret_0, G_ret, polar, Coulomb_mat, computation_name, n_chunks)
            dens_mat_iter = -2 * simps(G_ret_iter.imag*FDw, omega)
            correlated_di_iter = -1j * np.transpose( np.matmul( np.transpose( Sigma_GW_iter, axes=[2,0,1]), np.transpose( G_ret_iter, axes=[2,0,1]) ), axes=[1,2,0]) 
            correlated_di_iter = simps(correlated_di_iter * FDw, omega*2*np.pi)
            correlated_di_iter /= Uhu
            correlated_di_iter /= 2*np.pi
            total_double_occ_iter = 1/2 * np.concatenate( (np.diag(dens_mat_iter)[Nat:] * np.diag(dens_mat_MF)[0:Nat],   np.diag(dens_mat_MF)[Nat:] * np.diag(dens_mat_iter)[0:Nat]) )
            total_double_occ_iter -= np.diag( correlated_di_iter.real )   
            magn_moments_iter = np.diag(dens_mat_iter)[Nat:] + np.diag(dens_mat_iter)[0:Nat] - 2 * (total_double_occ_iter[0:Nat] + total_double_occ_iter[Nat:])
            with open('result-%i.txt' %(computation_name), 'a') as f:
                f.write( '\n' + '###############################################' + '\n')
                f.write('Magnetic moments: '  + '\n')
                f.write( '###############################################' + '\n')
            
            with open('result-%i.txt' %(computation_name), 'a') as f:
                f.write( str(magn_moments_iter) + '\n')   
                f.write( '###############################################' + '\n' )
                f.write( 'Sum magn_moments_iter : ' + str(np.sum(magn_moments_iter)) + '\n')   
        
        
        #If asked, we compute the absorption cross section
        if iteration == 1 and calc_abs==1:
            E_norm = 0.01
            nb_angle = 12
                
            def phi_tot_calc_hub(ind):
                epsilon = np.identity(len(EigenEnergies)) - xmul( One_over_r_Hartree, polar[:, :, ind])
                phi_tot = np.dot(inv(epsilon), phi_ext_mat_hub[:, ind])
                return phi_tot
                    
            def RPA_pol_1_r_GW(ind):
                epsilon = np.identity(len(EigenEnergies)) - xmul( One_over_r_Hartree, polar[:, :, ind])
                polar_RPA = xmul( polar[:, :, ind], inv(epsilon))
                return polar_RPA
                    
            ind_map = [ind for ind in range(len(omega))]
            polar_RPA_1_r = map(RPA_pol_1_r_GW, ind_map)                 
            polar_RPA_1_r = list(polar_RPA_1_r)
            polar_RPA_1_r = np.reshape(np.asarray(polar_RPA_1_r, dtype=complex), (len(polar_RPA_1_r), len(EigenEnergies), len(EigenEnergies)))

            phi_ext_mat = np.zeros((Nat, len(omega)))
            phi_ext_mat_hub = np.zeros((len(EigenEnergies), len(omega)))
            
            list_ind_angle = []
            list_angle = []
            gpi_MF_coll = []
            sigma_abs_coll = []
            sigma_abs_chi0_coll = []
            charges_coll = []
            for ind_angle in range(nb_angle+1):
              list_ind_angle.append(ind_angle)
            
            
            for ind_angle in range(nb_angle+1):
                angle = ind_angle * np.pi/nb_angle
                list_angle.append(angle)
                Ex_ext = E_norm * np.cos(angle)
                Ey_ext = E_norm * np.sin(angle)
                phi_ext = (- x * Ex_ext - y * Ey_ext) * (1.42/0.528)
                for ind in range(len(omega)):
                    phi_ext_mat[:,ind] = phi_ext[:] 
                for ind_w in range(len(omega)):
                    phi_ext_mat_hub[0:Nat, ind_w] = phi_ext_mat[:,ind_w]
                    phi_ext_mat_hub[Nat:, ind_w] = phi_ext_mat[:,ind_w]
                ind_map = [ind for ind in range(len(omega))]
                phi_tot = map(phi_tot_calc_hub, ind_map)
                phi_tot = list(phi_tot)
                phi_tot = np.reshape(np.asarray(phi_tot, dtype=complex), (len(phi_tot), len(EigenEnergies)))
                charges_MF = np.einsum( 'ijk, kj -> ik', polar, phi_tot)

                alpha = ( (-  np.einsum( 'kij, i, j -> k', polar_RPA_1_r, x_s, x_s, optimize=True  ) * Ex_ext - np.einsum( 'kij, i, j -> k', polar_RPA_1_r, y_s, y_s, optimize=True  ) * Ey_ext )) * ((1.42/0.528)**2) / E_norm
                sigma_abs = (4*np.pi) * 2*np.pi*omega * alpha.imag / light_vel
                sigma_abs_coll.append(sigma_abs)

                alpha = ( (-  np.einsum( 'ijk, i, j -> k', polar, x_s, x_s, optimize=True  ) * Ex_ext - np.einsum( 'ijk, i, j -> k', polar, y_s, y_s, optimize=True  ) * Ey_ext )) * (1.42/0.528)**2 / E_norm
                
                sigma_MF = (4*np.pi) * 2*np.pi*omega * alpha.imag / light_vel
                sigma_abs_chi0_coll.append(sigma_MF)
                
                gpi_MF = np.abs( np.einsum('ik, ki -> k', charges_MF, np.conj( phi_tot ), optimize=True ) ) / np.abs(  np.einsum('ik, ik -> k', charges_MF, np.conj( phi_ext_mat_hub ), optimize=True )  )
           
                gpi_MF_coll.append(gpi_MF)
                
            np.savez('MF absorption features - computation %i' %(computation_name), omega*2*np.pi*27.21, list_angle, sigma_abs_coll, sigma_abs_chi0_coll, gpi_MF_coll)


        #Mixing the Green's functions from iteration to iteration
        start_cond = time.time()
        if iteration >1:    
            G_ret_combili = alpha2 * G_ret_new + (1-alpha2-alphapr) * G_ret_new_old + alphapr * G_ret_new_old_old
            ecart = np.amax(np.abs(G_ret_new-G_ret_new_old))
            maxG = np.amax(np.abs(G_ret_new))
            ecart_rel = ecart/maxG
                    
        else:
            G_ret_combili = alpha2 * G_ret_new + (1-alpha2) * G_ret
            ecart = np.amax(np.abs(G_ret-G_ret_combili))
            maxG = np.amax(np.abs(G_ret_combili))
                     
        if iteration >1:
            G_ret_new_old_old = G_ret_new_old * 1
            G_ret_new_old = G_ret_new * 1
            
        else:
            G_ret_new_old_old = G_ret_new * 1       
            G_ret_new_old = G_ret_new * 1
    
        end_cond = time.time()
        end_loop = time.time() 
        with open('result-%i.txt' %(computation_name), 'a') as f: 
            f.write( 'Iteration time: ' + str(end_loop - start_loop) + '\n')    
                 
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('\n' + '\n' + '#################################################################################'  + '\n')
        f.write( '###############################################' + '\n')
        f.write('End of GW self-consistent cycle '  + '\n')
        f.write( '###############################################' + '\n') 
        f.write('#################################################################################'  + '\n')
        if iteration<GW_max_iteration:
            f.write('GW converged in %i iterations ' %(iteration)  + '\n')
            f.write('Convergence in energy: ' + str(np.abs(E_tot_gw-E_tot_gw_old))  + '\n' + '\n')
        else:
            f.write('GW did not converge in %i iterations ' %(iteration)  + '\n')
            f.write('Difference in energy: ' + str(np.abs(E_tot_gw-E_tot_gw_old))  + '\n' + '\n')
        
        
    #After GW self-consistency, we compute all GW physical quantities and output them in .npz files      
    dens_mat = -2 * simps(G_ret.imag*FDw, omega)
    H_mat_TB =  H_mat * 1
    np.fill_diagonal(H_mat_TB, 0.)
    Trace_G = -2 * np.trace( G_ret.imag, axis1=0, axis2=1)
    second_term_H_TB = np.trace( np.matmul(H_mat_TB, dens_mat)) /2
    E_gm_mu =  1/2 * simps(Trace_G * (omega + (FermiE_0+FermiE_before+FermiE_sec_keep)/(2*np.pi)) * FDw, omega) * 2 * np.pi
    
    q_sec_up_GW_simps = np.diag(dens_mat)[0 : Nat]
    q_sec_down_GW_simps = np.diag(dens_mat)[Nat:]
    q_sec_tot_GW_simps = q_sec_up_GW_simps + q_sec_down_GW_simps
    spin_GW_simps = (q_sec_up_GW_simps - q_sec_down_GW_simps)/2
    
    
    #Computing G and Sigma to get final double occupations
    G_ret_new, Sigma_GW_ret = Module_one_iter_GW_sigma.one_iter_GW_sigma(Nel, Nat, omega, omega_homo, EigenEnergies, FermiE_sec, kbT, bp, bm, G_ret_0, G_ret, polar, Coulomb_mat, computation_name, n_chunks)
      
    #correction of 19/04/23
    correlated_di = -1j * ( np.transpose( np.matmul( np.transpose( Sigma_GW_ret, axes=[2,0,1]), np.transpose( G_ret_new, axes=[2,0,1]) ), axes=[1,2,0])  - np.conj( np.transpose( np.matmul( np.transpose( G_ret_new, axes=[2,0,1]), np.transpose( Sigma_GW_ret, axes=[2,0,1]) ), axes=[2,1,0]) ) ) / 2 
    correlated_di = simps(correlated_di * FDw, omega*2*np.pi)
    correlated_di /= Uhu
    correlated_di /= 2*np.pi
    total_double_occ = 1/2 * np.concatenate( (np.diag(dens_mat)[Nat:] * np.diag(dens_mat_MF)[0:Nat],   np.diag(dens_mat_MF)[Nat:] * np.diag(dens_mat)[0:Nat]) )
    total_double_occ -= np.diag( correlated_di.real )
    
    np.savez('GW@MF correlated part double occupation - computation %i' %(computation_name), np.diag( correlated_di.real ) )
    np.savez('GW@MF double occupation - computation %i' %(computation_name), total_double_occ )
         
    #Some MF physical quantities and output them in .npz files       
    dos_MF = -2 * np.einsum('iik->k', G_ret_0.imag)
    L_dos_MF = -2 * np.einsum('iik->ik', G_ret_0.imag)
    total_double_occ_MF =  np.diag(dens_mat_MF)[0:Nat] * np.diag(dens_mat_MF)[Nat:]
    np.savez('MF double occupation - computation %i' %(computation_name), total_double_occ_MF )
    
    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write( '\n' + '###############################################' + '\n')
        f.write('Results summary: MF '  + '\n')
        f.write( '###############################################' + '\n') 
        f.write( 'Total spin MF: %f' %(spin_MF)  + '\n')
        f.write( 'Sum of absolute values of local spins: %f' %(magnetization_MF) + '\n')

    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write( '\n' + '###############################################' + '\n')
        f.write('Results summary: GW '  + '\n')
        f.write( '###############################################' + '\n') 
        f.write( 'Total spin GW: %f' %(np.sum(spin_GW_simps))  + '\n')
        f.write( 'Sum of absolute values of local spins: %f' %(np.sum(np.abs(spin_GW_simps))) + '\n')    
     
    ind_5 = int(np.where(np.abs(omega*2*np.pi*27.21-5) == np.amin(np.abs(omega*2*np.pi*27.21-5))) [0])
    ind_m5 = int(np.where(np.abs(omega*2*np.pi*27.21+5) == np.amin(np.abs(omega*2*np.pi*27.21+5))) [0])
    np.savez('MF DOS - computation %i' %(computation_name), omega*2*np.pi*27.21, dos_MF )
    np.savez('MF L DOS - computation %i' %(computation_name), omega*2*np.pi*27.21, L_dos_MF )
    np.savez('MF G_ret spin up - computation %i' %(computation_name), omega[ind_m5:ind_5]*2*np.pi*27.21, G_ret_0[0:Nat, 0:Nat, ind_m5:ind_5] )
    np.savez('MF G_ret spin down - computation %i' %(computation_name), omega[ind_m5:ind_5]*2*np.pi*27.21, G_ret_0[Nat:, Nat:, ind_m5:ind_5] )
 
    return G_ret, polar

  

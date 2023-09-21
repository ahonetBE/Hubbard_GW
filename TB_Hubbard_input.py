# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:37:41 2020
Main code for computing the GW approximation of the Hubbard model

The different parts of the code are the following:
    - inputs reading, parameters setting, ...
    - in the "if Hubbard" condition, the MF approximation of the Hubbard model is computed with the help of Modules "Module_TB_Hubbard_multi_hopping" and "Module_secant_Hubbard" as well as the function "Mean_field_Hubbard" doing the computation for one self-consistent iteration
    - in the "green_function" condition, we initiate the frequency grid and compute the GW Green's function with the help of the Module "Module_GW"
    - in the "calc_abs" condition, we compute the optical absorption cross section using the above-defined functions

@author: ahonet
"""



import numpy as np
import scipy.integrate as integrate
import time
from numpy.linalg import inv
from numpy.linalg import eigvals
import pybinding as pb
from pybinding.repository import graphene

import Module_secant_Hubbard
import Module_TB_Hubbard_multi_hopping
import Module_GW


t_begin=time.time()

#get file object
f = open("input_GW.txt", "r")
dictio = {}

while(True):
	#read next line
    line = f.readline()
	#if line is empty, you are done with all lines in the file
    if not line:
        break
    if line.startswith('#'):
        txt = line[1:].strip()
        continue
    if line.startswith('###'):
        continue
	#you can access the line
    dictio[txt] =  np.float(line)
#close file
f.close

#Accessing the inputs from the input file and putting them into variables
computation_name = int(dictio['computation_name'])
linear_chain  = int(dictio['linear_chain'])
number_atoms  = int(dictio['number_atoms'])
exp_2_gr  = int(dictio['exp_2_gr'])
ieta_exp = int(dictio['ieta_exp'])
mult_ieta = dictio['mult_ieta']

n_chunks = int(dictio['n_chunks'])
cond_shift = int(dictio['cond_shift'])

MF_toler = float(dictio['MF_toler'])
MF_max_it = int(dictio['MF_max_it'])
GW_toler = float(dictio['GW_toler'])
GW_max_iteration = int(dictio['GW_max_it'])

rand_state = int(dictio['rand_state'])
opposite_state = int(dictio['opposite_state'])
afm_state = int(dictio['afm_state'])
fm_state = int(dictio['fm_state'])
homogeneous_state = int(dictio['homogeneous_state'])
afm_chain_state = int(dictio['afm_chain_state'])

calc_abs = int(dictio['calc_abs'])

if rand_state == 1:
    with open('result-%i.txt' %(computation_name) , 'w') as f:
        f.write('Initial state = random state' + '\n')
elif opposite_state == 1:
    with open('result-%i.txt' %(computation_name) , 'w') as f:
        f.write('Initial state = opposite state' + '\n')
elif afm_state == 1:
    with open('result-%i.txt' %(computation_name) , 'w') as f:
        f.write('Initial state = afm state' + '\n')
elif fm_state == 1:
    with open('result-%i.txt' %(computation_name) , 'w') as f:
        f.write('Initial state = fm state' + '\n')
elif homogeneous_state == 1:
    with open('result-%i.txt' %(computation_name) , 'w') as f:
        f.write('Initial state = homogeneous state' + '\n')
elif afm_chain_state == 1:
    with open('result-%i.txt' %(computation_name) , 'w') as f:
        f.write('Initial state = afm chain state' + '\n')


###############################################################################
#Generation of the structure using Pybinding 
############################################################################### 
a = 0.24595   # [nm] unit cell length
a_cc = 0.142
#structure or for 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
N_cell = 1
with open('result-%i.txt' %(computation_name) , 'a') as f:
        f.write('\n'+ '#################################################################################'  + '\n')
        f.write('System: 5-AGNR of %i unit cells' %(N_cell) + '\n')
        f.write('#################################################################################'  + '\n')
width = 0.98*0.5
lengthgr = .36 * N_cell
model = pb.Model(
        graphene.monolayer(),
        pb.rectangle(x=1.5*width, y= 1.2*lengthgr).with_offset([0,15*a/2])
        #pb.rectangle(x=lengthgr, y= width) + pb.rectangle(x=1.5*lengthgr, y= 1.5*width).with_offset([10*a/2, 0]) + pb.rectangle(x=lengthgr, y= width).with_offset([20*a/2, 0])
        )


###############################################################################
#Generate the atoms coordinates
###############################################################################
x, y, z = model.system.x / a_cc, model.system.y / a_cc, model.system.z / a_cc

if linear_chain:
    Nat = number_atoms*1
    x = np.arange(Nat)
    y = np.zeros(Nat)
    z = np.zeros(Nat)
    with open('result-%i.txt' %(computation_name) , 'a') as f:
        f.write('\n'+ '#################################################################################'  + '\n')
        f.write('System replaced with a linear chain of %i atoms' %(number_atoms) + '\n')
        f.write('#################################################################################'  + '\n')
Nat = len(x)

z = np.zeros(len(x))

x_s = np.concatenate((x, x), axis=None)     
y_s = np.concatenate((y, y), axis=None) 
z_s = np.concatenate((z, z), axis=None) 



###############################################################################
#Some constants and 
#   structure parameters
##############################################################################

hopping = dictio['hopping']/27.21
hopping_2 = dictio['hopping_2']/27.21
hopping_3 = dictio['hopping_3']/27.21
delta_hopping = dictio['delta_hopping']/27.21
prop_u = dictio['prop_u']

NN_limit_dist = 1.1#16e-11
NN_limit_dist_2 = 1.8
NN_limit_dist_3 = 2.1
Nel = 0
on_site = 0 

Uhu		= prop_u * np.abs(hopping) 

with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('\n'+ '#################################################################################'  + '\n')
        f.write('Parameters of the model'  + '\n')
        f.write('#################################################################################'  + '\n')
with open('result-%i.txt' %(computation_name), 'a') as f:
    f.write('eta=' + str(mult_ieta * 10**(-ieta_exp) * 1j) + '\n')
    f.write('N freq=2^' + str(exp_2_gr)  + '\n' )
    f.write('bp=%.3f, bm=%.3f, fact_resol=%.3f, fact_resol_2=%.3f, lim_inf_resol=%.3f, lim_sup_resol=%.3f, lim_inf_resol_2=%.3f, lim_sup_resol_2=%.3f' %(dictio['bp'], dictio['bm'], dictio['fact_resol'], dictio['fact_resol_2'], dictio['lim_inf_resol'], dictio['lim_sup_resol'], dictio['lim_inf_resol_2'], dictio['lim_sup_resol_2'])  + '\n')
    f.write('t1 = ' + str(hopping*27.21) + ' eV' + '\t t2 = ' + str(hopping_2*27.21) + 'eV' + '\t  t3=' + str(hopping_3*27.21) + 'eV' + '\n  Delta t=' + str(delta_hopping*27.21) + 'eV' + '\t  U=' + str(Uhu*27.21) + 'eV' + '\t  U/t=' + str(Uhu/hopping) + '\n')
    f.write( '\n' + 'Condition on indices between Fermi levles for a shift (in GW):' + '\t' + '|indices between Fermi levels|' +  '>' + str(cond_shift) + '\n')
    f.write( 'MF convergence threshold: ' +  str(MF_toler) +  '\n' + 'MF maximum iterations: ' + str(MF_max_it) + '\n')
    f.write( 'GW convergence threshold: ' +  str(GW_toler) +  '\n' + 'GW maximum iterations: ' + str(GW_max_iteration) + '\n')
    
#on_site_dopant_s = np.zeros(2*Nat)    #We can define here the on-site term for special atoms (dopants). The index match the way the structure is constructed 
                                                # i.e. by section (1/6 of the structure) and by layers     
on_site_dopant = np.zeros(Nat)    #We can define here the on-site term for special atoms (dopants). The index match the way the structure is constructed 
                                                # i.e. by section (1/6 of the structure) and by layers 

N_free_radical = int(dictio['nb_free_radicals']) 
ind_free_radical = np.zeros(N_free_radical, dtype=int) 
for ind_rad in range(N_free_radical):
    ind_free_radical[ind_rad] = int(dictio['pos_free_radical_%i' %(ind_rad+1)])
    
free_rad_electron_minus = int(dictio['free_rad_electron_minus']) 
if free_rad_electron_minus  :
    Nel -=  N_free_radical
    

on_site_free_rad = dictio['on_site_free_rad'] / 27.21
on_site_dopant[ind_free_radical] = on_site_free_rad
if N_free_radical>0:
    on_site = on_site_free_rad*1
  
N_double_h = int(dictio['nb_double_h']) 
ind_double_h = np.zeros(N_double_h, dtype=int) 
for ind_dh in range(N_double_h):
    ind_double_h[ind_dh] = int(dictio['pos_double_h_%i' %(ind_dh+1)])


on_site_double_h = dictio['on_site_double_h'] / 27.21
on_site_dopant[ind_double_h] = on_site_double_h 
double_H_electron_plus = int(dictio['double_H_electron_plus']) 
if double_H_electron_plus  :
    Nel += N_double_h
if N_double_h>0:
    on_site = on_site_double_h*1

with open('result-%i.txt' %(computation_name), 'a') as f:
    f.write( '\n' + 'number of free radicals:' + str(N_free_radical) + '\n')
    f.write('free radical positions:' + str(ind_free_radical) + '\n')
    f.write('free radicals on-site value:' + str(on_site_dopant[ind_free_radical]*27.21) + 'eV' + '\n')
    f.write('removing an electron per free radical? :' + str(bool(free_rad_electron_minus))  + '\n')
    f.write('\n' + 'number of double H:' + str(N_double_h) + '\n')
    f.write('free double H positions:' + str(ind_double_h) + '\n')
    f.write('double H on-site value:' + str(on_site_dopant[ind_double_h]*27.21) + 'eV' + '\n')
    f.write('adding an electron per free radical? :' + str(bool(double_H_electron_plus)) + '\n')

q_00 = np.ones(Nat)  
q_00[np.nonzero(on_site_dopant)] += 1

light_vel = 3e8/(2.18e6)              
hbar = 1
kbT = 0.025/27.21  #in Hartree
on_site_eps = 0.
###############################################################################
#Applied electric field
##############################################################################
Ex_ext = 0.01   # = 5.14e7 V/cm = 5.14 V/nm in Hartree units 
Ey_ext = 0.01
phi_ext = (- x * Ex_ext - y * Ey_ext) * (1.42/0.528)
E_norm = np.sqrt( np.square(Ex_ext) + np.square(Ey_ext) ) 


with open('result-%i.txt' %(computation_name), 'a') as f:
    f.write('\n' + '\n' + '#################################################################################'  + '\n')
    f.write( '###############################################' + '\n')
    f.write('MF self-consistent cycles start'  + '\n')
    f.write('###############################################'  + '\n')
    f.write('#################################################################################' +'\n' + '\n')
def Mean_field_Hubbard():
    ###############################################################################
    #Use the TB_Hubbard module for the positions of the atoms calculated
    #   It is made in a self-consistent way with the while loop
    ###############################################################################
    q_sec0 = np.zeros(2*Nat)
    q_sec = np.ones(2*Nat)
    if rand_state ==1:
        q_sec = np.random.random(2*Nat)
    elif opposite_state == 1:
        q_sec[0:Nat] = np.ones(Nat)
        q_sec[Nat:] = -np.ones(Nat)
    elif afm_state == 1:
        half_Nat = int(Nat/2)
        q_sec[0:half_Nat] = np.ones(half_Nat)
        q_sec[half_Nat:Nat] = -np.ones(half_Nat)
        q_sec[Nat:] = - q_sec[0:Nat]
    elif fm_state == 1:
        q_sec[0:Nat] = np.ones(Nat)
        q_sec[Nat:] = np.zeros(Nat)
    elif homogeneous_state == 1:
        q_sec[0:Nat] = np.ones(Nat)
        q_sec[Nat:] = np.ones(Nat)
    elif afm_chain_state == 1:
       q_sec[0:Nat:2] = 1
       q_sec[1:Nat:2] = -1
       q_sec[Nat:2*Nat:2] = -1
       q_sec[1+Nat:2*Nat:2] = 1
            
            
    sum_q_sec = np.sum(np.abs(q_sec))
    q_sec *= (1 * Nat + Nel) / sum_q_sec
    borne1_EF = -0.#-1.5
    borne2_EF = 0.3
    tol=1e-7
    max_it=5000
    it=0
    alpha=0.5#0.2#0.75
    while ( (np.amax(np.abs(q_sec-q_sec0))>MF_toler) and (it<MF_max_it) ):
        it+=1
                
        q_sec0 = (1-alpha) * q_sec + alpha * q_sec0
        if it == 1:
            q_sec0 = q_sec * 1
        EigenEnergies, EigenVectors, H_mat = Module_TB_Hubbard_multi_hopping.Tight_Binding_Hubbard(Nat, x, y, z, NN_limit_dist, NN_limit_dist_2, NN_limit_dist_3, on_site_eps, on_site_dopant, hopping, hopping_2, hopping_3, delta_hopping, q_sec0, Uhu)        
        FermiE_sec, q_sec = Module_secant_Hubbard.secant_method(borne1_EF ,borne2_EF, tol, max_it, Nat, EigenVectors, EigenEnergies, Nel, kbT)  
                
        while np.isnan(FermiE_sec) :
            borne1_EF = -np.abs(np.random.random()) + tol * 1.1
            borne2_EF = borne1_EF +  np.abs(np.random.random()) + tol * 1.1
            FermiE_sec, q_sec = Module_secant_Hubbard.secant_method(borne1_EF ,borne2_EF, tol, max_it, Nat, EigenVectors, EigenEnergies, Nel, kbT)  
        
               
        if np.any(np.isnan(q_sec)) or np.any(np.isinf(q_sec)):
            q_sec = np.random.random(2*Nat)
            sum_q_sec = np.sum(q_sec)
            q_sec *= (2 * Nat + Nel) / sum_q_sec
        
    global FermiE_sec_tot; FermiE_sec_tot = 0
    EigenEnergies -= FermiE_sec
    FermiE_sec_tot += FermiE_sec
    FermiE_sec = 0.
      
        
    if Nel ==0: 
        FermiE_sec = (EigenEnergies[Nat] + EigenEnergies[Nat-1])/2
        EigenEnergies -= FermiE_sec
        FermiE_sec_tot += FermiE_sec
        FermiE_sec = 0.
    else:
        min_pos = min(i for i in EigenEnergies if i > 0)
        max_neg = max(i for i in EigenEnergies if i < 0)
        FermiE_sec = (EigenEnergies[np.where(EigenEnergies == max_neg)] + EigenEnergies[np.where(EigenEnergies == min_pos)])/2
        FermiE_sec = np.amin(FermiE_sec)
        EigenEnergies -= FermiE_sec
        FermiE_sec_tot += FermiE_sec
        FermiE_sec = 0.
        
    Total_Energy = np.sum( EigenEnergies /(np.exp((EigenEnergies-FermiE_sec)/kbT)+1) ) 
    Energy_correction = Uhu * np.sum(q_sec[0 : Nat]*q_sec[Nat :])
    Hubbard_energy = (Total_Energy-Energy_correction)
    Hubbard_E_plus_muN = Hubbard_energy + FermiE_sec_tot*(Nat+Nel)
        
    q_tot = q_sec[0 : Nat] + q_sec[Nat :]
    sum_Q = np.sum(q_tot)
    spin_z = (q_sec[0 : Nat] - q_sec[Nat :]) / 2
    sum_spin = np.sum(spin_z)

    with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('\n' + '###############################################' + '\n')
        f.write('New MF cycle' + '\n')
        f.write('###############################################' + '\n')
        f.write('Energy Tot (eV): ' + str((Hubbard_energy+FermiE_sec_tot*(Nat+Nel))*27.21) + '\n')
        f.write('Fermi level (eV): ' + str(FermiE_sec*27.21) + '\n')
        if it<MF_max_it:
            f.write('MF converged with %i iterations ' %(it) + '\n')
            f.write('Convergence in charges: ' + str(np.amax(np.abs(q_sec-q_sec0))) + '\n')
        else :
            f.write('MF did not converge with %i iterations' %(it) + '\n')
            f.write('Convergence in charges: ' + str(np.amax(np.abs(q_sec-q_sec0))) + '\n')
        f.write('MF total charge: ' + str(sum_Q) + '\n')
        f.write('MF total spin: ' + str(sum_spin) + '\n')
            
    return EigenEnergies, EigenVectors, FermiE_sec, q_sec, Hubbard_E_plus_muN, H_mat, sum_spin
        
EigenEnergies, EigenVectors, FermiE_sec, q_sec, Hubbard_energy, H_mat, sum_spin = Mean_field_Hubbard()
interation_MF = 0
Hubbard_energy_min = 0
limite_iter = 50
limite_inf = 3
diff_energy = 1
    
#This is the MF self-consistent loop, using the function "Mean_field_Hubbard()" for single-iteration computation 
while(diff_energy > 1e-6 or interation_MF <= limite_inf):
    interation_MF += 1 
    EigenEnergies, EigenVectors, FermiE_sec, q_sec, Hubbard_energy, H_mat, sum_spin = Mean_field_Hubbard() 
    diff_energy = Hubbard_energy-Hubbard_energy_min
        
    if Hubbard_energy < Hubbard_energy_min:
        Hubbard_energy_min = Hubbard_energy*1
    if interation_MF > limite_iter:
        break
    if homogeneous_state and (sum_spin > 1e-5):
        diff_energy = 1

with open('result-%i.txt' %(computation_name), 'a') as f:
    f.write('\n' +'\n' + '#################################################################################'  + '\n')
    f.write( '###############################################' + '\n')
    f.write('End of MF cycles' + '\n')
    f.write('###############################################' + '\n')
    f.write('#################################################################################'  + '\n' + '\n')
    f.write('Number of MF self-consistent cycles: ' + str(interation_MF+1) + '\n')         
    f.write('Minimum total energy found: ' + str(Hubbard_energy_min*27.21) + '\n')
    f.write('Last total energy: ' + str(Hubbard_energy*27.21) + '\n')
    
    

ieta = mult_ieta * 10**(-ieta_exp) * 1j
    
#Initiate the frequency grid
bp = float(dictio['bp'])* np.abs(hopping)
bm = -float(dictio['bm'])* np.abs(hopping)
                
length = 2**exp_2_gr*2                 #number of frequencies
omega = np.linspace(bm,bp,num=length)
                
fact_resol = int(dictio['fact_resol'])   
fact_resol_2 = int(dictio['fact_resol_2'])   
                
lim_inf_resol = -float(dictio['lim_inf_resol']) * np.abs(hopping)
lim_sup_resol = float(dictio['lim_sup_resol']) * np.abs(hopping)
                
lim_inf_resol_2 = -float(dictio['lim_inf_resol_2']) * np.abs(hopping)
lim_sup_resol_2 = float(dictio['lim_sup_resol_2']) * np.abs(hopping)
               
ind_lim_inf_resol_2 = int(np.where(np.abs(omega-lim_inf_resol_2) == np.amin(np.abs(omega-lim_inf_resol_2))) [0][0])
ind_lim_sup_resol_2 = int(np.where(np.abs(omega-lim_sup_resol_2) == np.amin(np.abs(omega-lim_sup_resol_2))) [0][0])
             
ind_lim_inf_resol = int(np.where(np.abs(omega-lim_inf_resol) == np.amin(np.abs(omega-lim_inf_resol))) [0][0])
ind_lim_sup_resol = int(np.where(np.abs(omega-lim_sup_resol) == np.amin(np.abs(omega-lim_sup_resol))) [0][0])
             
omega12 = omega[0:ind_lim_inf_resol_2]
omega22 = omega[ind_lim_inf_resol_2:ind_lim_inf_resol]
omega32 = omega[ind_lim_inf_resol:ind_lim_sup_resol]
omega42 = omega[ind_lim_sup_resol:ind_lim_sup_resol_2]
omega52 = omega[ind_lim_sup_resol_2:]
omega12 = omega12[0::fact_resol]
omega22 = omega22[0::fact_resol_2]
omega42 = omega42[0::fact_resol_2]
omega52 = omega52[0::fact_resol]
                
omega_red = np.concatenate((omega12, omega22, omega32, omega42, omega52), axis=0)
                
#Compute the GW Green's function in the module "Module_GW"
G_ret_GW, polar_GW = Module_GW.calc_GW(x, y, z, Uhu, prop_u, on_site, Nel, bp, bm, Nat, omega_red, omega, FermiE_sec, kbT, EigenVectors, EigenEnergies, ieta, H_mat, FermiE_sec_tot, calc_abs, computation_name, cond_shift, GW_toler, GW_max_iteration, n_chunks)   
dos_GW = -2 * np.einsum('iik->k', G_ret_GW.imag)
L_dos_GW = -2 * np.einsum('iik->ik', G_ret_GW.imag)

ind_5 = int(np.where(np.abs(omega_red*2*np.pi*27.21-5) == np.amin(np.abs(omega_red*2*np.pi*27.21-5))) [0])
ind_m5 = int(np.where(np.abs(omega_red*2*np.pi*27.21+5) == np.amin(np.abs(omega_red*2*np.pi*27.21+5))) [0])
                
np.savez('GW@MF DOS - computation %i' %(computation_name), omega_red*2*np.pi*27.21, dos_GW )
np.savez('GW@MF L DOS - computation %i' %(computation_name), omega_red*2*np.pi*27.21, L_dos_GW )
                
np.savez('GW@MF G_ret spin up - computation %i' %(computation_name), omega_red[ind_m5:ind_5]*2*np.pi*27.21, G_ret_GW[0:Nat, 0:Nat, ind_m5:ind_5] )
np.savez('GW@MF G_ret spin down - computation %i' %(computation_name), omega_red[ind_m5:ind_5]*2*np.pi*27.21, G_ret_GW[Nat:, Nat:, ind_m5:ind_5] )
                
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
                

     
#Define some functions useful for the absorption cross section computation
def xmul(a, b):
    out = np.empty_like(a)
    for j in range(a.shape[0]):
        out[j] = np.dot(a[j], b[j])
    return out    

def RPA_pol_1_r_GW(ind):
    epsilon = np.identity(len(EigenEnergies)) - xmul( One_over_r_Hartree, polar_GW[:, :, ind])
    polar_RPA = xmul( polar_GW[:, :, ind], inv(epsilon))
    return polar_RPA

def phi_tot_calc_hub(ind):
    epsilon = np.identity(len(EigenEnergies)) - xmul( One_over_r_Hartree, polar_GW[:, :, ind])
    phi_tot = np.dot(inv(epsilon), phi_ext_mat_hub[:, ind])
    return phi_tot


#Computation of the optical absorption cross section
if calc_abs == 1:

  ind_map = [ind for ind in range(len(omega_red))]
  polar_RPA_1_r = map(RPA_pol_1_r_GW, ind_map)                 
  polar_RPA_1_r = list(polar_RPA_1_r)
  polar_RPA_1_r = np.reshape(np.asarray(polar_RPA_1_r, dtype=complex), (len(polar_RPA_1_r), len(EigenEnergies), len(EigenEnergies)))            
  phi_ext_mat = np.zeros((Nat, len(omega_red)))
  phi_ext_mat_hub = np.zeros((len(EigenEnergies), len(omega_red)))
  nb_angle = 12
  E_norm = 0.01
  gpi_GW_coll = []
  sigma_abs_coll = []
  sigma_abs_chi0_coll = []
  charges_coll = []
  list_ind_angle = []
  list_angle = []
  
  for ind_angle in range(nb_angle+1):
      list_ind_angle.append(ind_angle)
      angle = ind_angle * np.pi/nb_angle
      list_angle.append(angle)
      Ex_ext = E_norm * np.cos(angle)
      Ey_ext = E_norm * np.sin(angle)
      phi_ext = (- x * Ex_ext - y * Ey_ext) * (1.42/0.528)
      phi_ext_mat = np.zeros((Nat, len(omega_red)))
      for ind in range(len(omega_red)):
          phi_ext_mat[:,ind] = phi_ext[:] 
      for ind_w in range(len(omega_red)):
          phi_ext_mat_hub[0:Nat, ind_w] = phi_ext_mat[:,ind_w]
          phi_ext_mat_hub[Nat:, ind_w] = phi_ext_mat[:,ind_w]
                 
      ind_map = [ind for ind in range(len(omega_red))]
      phi_tot = map(phi_tot_calc_hub, ind_map)
      phi_tot = list(phi_tot)
      phi_tot = np.reshape(np.asarray(phi_tot, dtype=complex), (len(phi_tot), len(EigenEnergies)))
      charges = np.einsum( 'ijk, kj -> ik', polar_GW, phi_tot)
  
      alpha = ( (-  np.einsum( 'kij, i, j -> k', polar_RPA_1_r, x_s, x_s, optimize=True  ) * Ex_ext - np.einsum( 'kij, i, j -> k', polar_RPA_1_r, y_s, y_s, optimize=True  ) * Ey_ext )) * ((1.42/0.528)**2) / E_norm
      sigma_abs = (4*np.pi) * 2*np.pi*omega_red * alpha.imag / light_vel 
  
      sigma_abs_coll.append(sigma_abs)

      alpha = ( (-  np.einsum( 'ijk, i, j -> k', polar_GW, x_s, x_s, optimize=True  ) * Ex_ext - np.einsum( 'ijk, i, j -> k', polar_GW, y_s, y_s, optimize=True  ) * Ey_ext )) * (1.42/0.528)**2 / E_norm
      sigma_abs = (4*np.pi) * 2*np.pi*omega_red * alpha.imag / light_vel
      
      sigma_abs_chi0_coll.append(sigma_abs)

      gpi_GW = np.abs( np.einsum('ik, ki -> k', charges, np.conj( phi_tot ), optimize=True ) ) / np.abs(  np.einsum('ik, ik -> k', charges, np.conj( phi_ext_mat_hub ), optimize=True )  )
      gpi_GW_coll.append(gpi_GW)
      
  np.savez('GW@MF absorption features - computation %i' %(computation_name), omega_red*2*np.pi*27.21, list_angle, sigma_abs_coll, sigma_abs_chi0_coll, gpi_GW_coll)
   
t_end = time.time()-t_begin

with open('result-%i.txt' %(computation_name), 'a') as f:
        f.write('\n' + 'Final time: ' + str(t_end) + '\n')


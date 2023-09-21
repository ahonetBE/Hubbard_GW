# Hubbard_GW inputs' and outputs' GUIDE
This document is a quick guide for the user that describes the formats of the input file and the ouput files. The variables in the input file are also detailed here.

The guide is divided in 3 parts: 
* general guide for the input.txt file
* a detailed parameter guide 
* a guide for ouput files' formats.

## General guide for input.txt: 

* each line starting with a single "#" sign specifies the name of input variable to be entered the following line

* each line starting directly with a number specify the value of the variable named the line before

* each line starting with three "#" signs are ignored by the code

## Parameters guide/list (in the input.txt file) :

* computation_name: integer; asign an identifier to the computation. All output files will be named according to this number.
* linear_chain: 1 or 0; if linear_chain=1 the model defined in the main file using pybinding will be replaced by a linear chain.
* number_atoms: integer; if linear_chain=1, number_atoms is length of the linear chain to consider.
* exp_2_gr: integer; the number of discretized frequencies in the grid. This number is 2 to the power of exp_2_gr.
* bp and bm: floats; respectively the upper and lower boundaries of the frequency grid. The boundaries are given by 2*pi*bp ???
* lim_sup_resol and lim_inf_resol: floats; values outside which the frequency grid is less dense for the evaluation of . The frequency grid is less dense by a factor of fact_resol.
* lim_sup_resol_2, lim_inf_resol_2 and fact_resol_2: the same as lim_sup_resol, lim_inf_resol and fact_resol except that the frequency grid is another time less dense. Therefore, lim_sup_resol_2 has to be greater than lim_sup_resol and lim_inf_resol_2 has to be smaller than lim_inf_resol.
* ieta_exp and mult_ieta: integers; define the value of the eta parameter involved in the numerical evaluation of the Green's function. eta is given by mult_ieta * 10^ieta_exp.
* n_chunks: integer; defines the numbet of chunks used in evaluation of the Green's function, the polarizability and self-energy. n_chunks has to be a diviser of 2*Nat where Nat is the number of atoms in the structure.
* cond_shift: integer; the Fermi levels of the Green's functions will be aligned for each GW iteration if the number of frequencies in the frequency grid between both Fermi levels is smaller than cond_shift. Setting cond_shift=0, the user chooses to always align the Fermi levels while setting a very large value for cond_shift, the probability to align them is rather small. The user can run a first test to sea the order of magnitude of the difference between GW iterations and then choose an appropriate value of cond_shift to force the code not the align Fermi levels.
* MF_toler: float; defines the threshold for convergence for MF computations. The MF computations are considered to be converged if the maximum of the difference between charges in two successive iterations (in absolute value) is smaller than MF_toler.
* MF_max_it: integer; maximum number of iteration during MF self-consistence cycles. If it is reached, the code exits the cycle without convergence achieved.
* GW_toler: float; threshold for convergence for GW computations. The GW computations are considered to be converged if the difference between energies in two successive iterations (in absolute value) is smaller than GW_toler
* GW_max_it: integer; maximum number of iteration during GW self-consistence cycles. If it is reached, the code exits the cycle without convergence achieved.
* rand_state: 1 or 0; if rand_state = 1, the initial guess for densities in MF self-consistency is set to random values between 0 and 1. 
* opposite_state: 1 or 0; to be used for graphene nanosystems or linear chains. If opposite_state = 1 (and rand_state=0), the initial guess for densities in MF self-consistency is set +1 for spin up sector and -1 for spin down sector. 
* afm_state: 1 or 0; to be used for graphene nanosystems only. If afm_state = 1 (and rand_state=0, opposite_state=0), the initial guess for densities in MF self-consistency is set +1 for spin up sector on sublattice A and -1 for spin up sector on sublattice B. The opposite values are used for spin down sector. 
* fm_state: 1 or 0; to be used for graphene nanosystems only. If fm_state = 1 (and rand_state=0, opposite_state=0, afm_state=0), the initial guess for densities in MF self-consistency is set +1 for spin up sector on sublattice A and -1 for spin down sector. 
* homogeneous_state: 1 or 0; to be used for graphene nanosystems or linear chains. If homogeneous_state = 1 (and rand_state=0, opposite_state=0, afm_state=0, fm_state=0), the initial guess for densities in MF self-consistency is set +1 for both spin up and spin down on all sites.
* afm_state_chain: 1 or 0; to be used for linear chains only. If afm_state_chain = 1 (and rand_state=0, opposite_state=0, afm_state=0, fm_state=0, homogeneous_state=0), the initial guess for densities in MF self-consistency is alternatively set to +1 and -1 for spin up and the opposite values for spin down.
* hopping: float; value of nearest-neighbour hopping parameter
* hopping_2: float; value of second nearest-neighbour hopping parameter. Could be non-zero only for graphene nanosystems, not for linear chains.
* hopping_3: float; value of third nearest-neighbour hopping parameter. Could be non-zero only for graphene nanosystems, not for linear chains.
* delta_hopping: float; value of change in nearest-neighbour hopping parameter for atoms at the edges of graphene.
* prop_u: float; value of the ratio U/t (interaction over hopping parameter)
* calc_abs: 1 or 0; if calc_abs=1, the optical absorption cross section and generalized plasmonic index will be computed. If calc_abs=0, they won't be.
* nb_free_radicals: integer; specifies the number of modified atomic sites of type "free_radicals". They are associated to an on-site potential and the removing of an electron or not.
* free_rad_electron_minus: 1 or 0; if free_rad_electron_minus=1, nb_free_radicals electrons will be removed from half-filling.
* on_site_free_rad: set the on-site potential of modified atomic sites of type "free_radicals".
* pos_free_radical_1, pos_free_radical_2, ... : integers between 0 and Nat-1 where Nat is the number of atoms. It specifies the indices of modified atomic sites of type "free_radicals" as indexed using pybinding (for graphene nanostructures) or indexed in increasing order in linear chains. The user can add lines "pos_free_radical_3", ..., "pos_free_radical_n" and the nb_free_radicals first indices will be considered by the code.
* nb_double_h: integer; specifies the number of modified atomic sites of type "double_h". They are associated to an on-site potential and the addition of an electron or not.
* double_H_electron_plus: 1 or 0; if double_H_electron_plus=1, nb_double_h electrons will be removed from half-filling.
* on_site_double_h: set the on-site potential of modified atomic sites of type "double_h".
* pos_double_h_1, pos_double_h_2, ... : integers between 0 and Nat-1 where Nat is the number of atoms. It specifies the indices of modified atomic sites of type "double_h" as indexed using pybinding (for graphene nanostructures) or indexed in increasing order in linear chains. The user can add lines "pos_double_h_3", ..., "pos_double_h_n" and the nb_double_h first indices will be considered by the code.


## Output files guide:
For a single run of the code, 20 output files are generated with .npz format. For the output files, there are files starting with MF and GW@MF for MF and GW approximations but also starting by G0W0@MF that are related to the results after the first GW iteration (this is a widely used approximation in the ab initio community doing dynamical screening). Pay attention that the Green's functions in .npz files are not savez with the same frequency grid than the other quantities for the sake of memory saving.
In this section and for output files, XXX will refer to the computation_name defined in the input.py file.

* DOS files are named: "MF DOS - computation XXX.npz", "G0W0@MF DOS - computation XXX.npz" or "GW@MF DOS - computation XXX.npz". They contain two quantities that are numpy arrays: "arr_0" is the frequency array while "arr_1" is the DOS computed at the corresponding frequencies.
* LDOS files are named: "MF L DOS - computation XXX.npz", "G0W0@MF L DOS - computation XXX.npz" or "GW@MF L DOS - computation XXX.npz". They contain two quantities that are "arr_0" (frequency array) and "arr_1" a two-dimensional array, with first index the site/spin index and the second index the frequency index (corresponding to the frequencis in "arr_0"). The first index range from 0 to 2*Nat-1 and indices from 0 to Nat-1 correspond to a given spin ordered using the Pybinding's indices. The same olds for Nat to 2*Nat-1 indices, corresponding to the opposite spin.
* G_ret files are named: "MF G_ret spin YYY - computation XXX.npz", "G0W0@MF G_ret spin YYY - computation XXX.npz" or "GW@MF G_ret spin YYY - computation XXX.npz" where YYY is either "up" or "down". They contain two quantities that are "arr_0" (frequency array that is different from the DOS/L DOS frequencies) and "arr_1" a three-dimensional array. Its two first indices range from 0 to Nat-1 and correspond to atomic indices as ordered by the Pybinding model and the third index corresponds to the frequency index, related to the frequency grid saved in the same file as "arr_0".
* Double occupation files are named: "MF double occupation - computation XXX.npz", "G0W0@MF double occupation - computation XXX.npz" or "GW@MF double occupation - computation XXX.npz". They contain one quantity that is a one-dimensional numpy array of length 2*Nat. The double occupations are obtained summing the Nat first terms and the Nat last terms one-to-one (e.g. the first double occupation is the sum of the elements 0 and Nat of the array). They correspond to double occupations on atomic sites ordered as in the Pybinding model.
* Correlated part of double occupation files are named: "G0W0@MF correlated part double occupation - computation XXX.npz" or "GW@MF correlated part double occupation - computation XXX.npz". There is not MF file since the correlated part of the double occupations are zero. They contain one quantity that is a one-dimensional numpy array of length 2*Nat. The correlated part of the double occupations are obtained summing the Nat first terms and the Nat last terms one-to-one (e.g. the first double occupation is the sum of the elements 0 and Nat of the array). They correspond to correlated part of double occupations on atomic sites indexed as in the Pybinding model.

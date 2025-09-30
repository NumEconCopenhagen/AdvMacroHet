import time
import numpy as np
from consav.grids import equilogspace
from consav.misc import elapsed
from scipy.optimize import root
from tauchen import log_tauchen_nb

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    
    # b. z
    par.z_log_grid,par.z_grid[:],z_trans,par.z_ergodic[:] = log_tauchen_nb(par.rho_z,par.sigma_psi,n=par.Nz)
    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = z_trans
        par.z_trans_ss[i_fix,:,:] = z_trans

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    ss.z_trans[0,:,:] = z_trans
    ss.Dbeg[0,:,0] = par.z_ergodic
    ss.Dbeg[0,:,1:] = 0.0 

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]
            income = ss.w**z

            c = (1+ss.r)*par.a_grid + income
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]
    
def obj_ss(x,model,method,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    if method == 'direct':

        if ss.upsilon == 1.0:
            ss.K = x[0]
            ss.z_scale = 1.0
        else:
            ss.K, ss.z_scale = x

        if do_print:
            print(f'guess {ss.K = :.4f}')    

    elif method == 'beta':

        if ss.upsilon == 1.0:
            par.beta = x[0]
            ss.z_scale = 1.0
        else:
            par.beta, ss.z_scale = x
        
        if do_print:
            print(f'guess {par.beta = :.4f}')

    # a. production    
    ss.Gamma = par.Gamma_ss
    ss.Y = ss.Gamma*ss.K**par.alpha

    # b. implied prices
    ss.rK = par.alpha*ss.Y / ss.K 
    ss.r = ss.rK - par.delta
    ss.w = (1.0-par.alpha)*ss.Y
    ss.I = par.delta*ss.K
    ss.A = ss.K
    ss.upsilon = par.upsilon_ss
    
    if do_print:
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')
        
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # c. market clearing
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I
    ss.clearing_Z = ss.Z_hh-1.0

    if ss.upsilon == 1.0:
        if do_print: print(f'discrepancy in A = {ss.clearing_A:12.8f}') 
        if do_print: print(' ')
        return ss.clearing_A # target to hit
    else:
        if do_print: print(f'discrepancy in A = {ss.clearing_A:12.8f}') 
        if do_print: print(f'discrepancy in Z = {ss.clearing_Z:12.8f}')
        if do_print: print(' ')
        return np.array([ss.clearing_A, ss.clearing_Z]) # target to hit
    
def find_ss(model,method='direct',do_print=False,beta_guess = 0.9, K_guess = 4.0, z_scale_guess = 1.0):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,K_guess=K_guess,z_scale_guess=z_scale_guess)
    elif method == 'beta':
        find_ss_beta(model,beta_guess=beta_guess,do_print=do_print)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,do_print=False,K_guess=4.0,z_scale_guess=1.0):
    """ find steady state using direct method """

    if model.par.upsilon_ss == 1.0:
        model.ss.upsilon = model.par.upsilon_ss
        if do_print: print('solving for K only')
        res = root(obj_ss, x0 = K_guess, args=(model,'direct',do_print), method='hybr')
    else:
        model.ss.upsilon = model.par.upsilon_ss
        if do_print: print('solving for K and z_scale')
        res = root(obj_ss, x0 = (K_guess, z_scale_guess), args=(model,'direct',do_print), method='hybr')

    if do_print:
        print(f'Implied K = {model.ss.K:6.3f}')
        print(f'Implied Y = {model.ss.Y:6.3f}')
        print(f'Implied Gamma = {model.ss.Gamma:6.3f}')
        print(f'Implied delta = {model.par.delta:6.3f}') # check is positive
        print(f'Implied K/Y = {model.ss.K/model.ss.Y:6.3f}') 
        print(f'Discrepancy in A = {model.ss.clearing_A:12.8f}') 
        print(f'Discrepancy in Y = {model.ss.clearing_Y:12.8f}') # != 0 due to numerical error

def find_ss_beta(model,beta_guess=0.5,z_scale_guess=1.0, do_print=False):
    """ find steady state using indirect method """

    par = model.par
    ss = model.ss

    ss.r = par.r_ss_target
    ss.w = par.w_ss_target
    ss.K = ss.A = par.K_ss_target 
    ss.Y = par.Y_ss_target
    par.Gamma_ss = ss.Gamma = ss.Y / ss.K**par.alpha
    ss.rK = par.alpha*ss.Y / ss.K
    par.delta = ss.rK - ss.r

    if model.par.upsilon_ss == 1.0:
        model.ss.upsilon = model.par.upsilon_ss
        if do_print: print('solving for beta only')
        res = root(obj_ss, x0 = beta_guess, args=(model,'beta',do_print), method='hybr')
    else:
        model.ss.upsilon = model.par.upsilon_ss
        if do_print: print('solving for beta and z_scale')
        res = root(obj_ss, x0 = (beta_guess, z_scale_guess), args=(model,'beta',do_print), method='hybr')

    if do_print:
        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied Gamma = {ss.Gamma:6.3f}')
        print(f'Implied delta = {par.delta:6.3f}') # check is positive
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}') 
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}') # != 0 due to numerical error 

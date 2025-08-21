# find steady state

import time
import numpy as np
from scipy import optimize
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def aggregate_to_annual(quarterly_data):
    # Ensure the quarterly data length is a multiple of 4 (4 quarters per year)
    if len(quarterly_data) % 4 != 0:
        raise ValueError("The length of the quarterly data must be a multiple of 4.")
    
    # Reshape the array into a 2D array where each row is a year (4 quarters per row)
    quarterly_data = np.array(quarterly_data)
    reshaped_data = quarterly_data.reshape(-1, 4)
    
    # Sum the quarters for each year
    annual_data = reshaped_data.sum(axis=1)
    
    return annual_data



def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. z
    par.z_grid[:],ss.z_trans[:,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,0] = e_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]
            income = ss.w*ss.L*z+ss.chi

            c = (1+ss.ra)*par.a_grid + income
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.chi = 0.0
    ss.L = 1.0
    ss.pi = ss.pi_w = 0.0
    ss.eps_i = 0.0
    
    # c. monetary policy
    ss.ra = ss.i = ss.r = par.r_target_ss

    # d. firms
    par.Gamma = 1. 
    ss.Y = par.Gamma * ss.L 
    ss.w = par.Gamma/par.mu 
    ss.Z = ss.w*ss.L

    # e. government
    ss.chi = ss.r*ss.B

    # f. household 
    model.solve_hh_ss(do_print=False)
    model.simulate_hh_ss(do_print=False)

    # g. market clearing
    ss.Div = ss.Y - ss.w*ss.L 
    ss.pD = ss.Div / ss.r 

    ss.clearing_A = ss.A_hh -  ss.pD - ss.B 
    ss.clearing_Y = ss.Y-ss.C_hh

    # h. NK wage curve
    par.varphi = (1/par.mu*ss.w*ss.C_hh**(-par.sigma))/ss.L**par.nu
    ss.NKWC_res = 0.0 # used to derive par.varphi

def obj_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    if par.do_B:
        par.mu = 1.0
        ss.B = x[0]
        par.beta = x[1]

    else:

        ss.B = 0. 
        par.mu = x[0]
        par.beta = x[1]

    evaluate_ss(model,do_print=do_print)
    
    model._compute_jac_hh(inputs_hh_all=['chi'])
    ann_mpcs = aggregate_to_annual(-model.jac_hh[('C_hh','chi')][:,0])

    residual = np.array([ss.clearing_A, ann_mpcs[0] - 0.5]) 


    return residual 

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    if par.do_B:

        par.mu = 1.0
        ss.B = 0.7 
        par.beta = 0.988    
        x0 = np.array([ss.B,par.beta])

    else:

        par.mu = 1.007780
        par.beta = 0.988
        x0 = np.array([par.mu,par.beta])

    sol = optimize.root(obj_ss, x0, method='hybr', args=(model,do_print))

    # b. print
    if do_print:

        print(f' Y = {ss.Y:8.4f}')
        print(f' r    = {ss.r:8.4f}')
        print(f' A    = {ss.A_hh:8.4f}')
        print(f' pD    = {ss.pD:8.4f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
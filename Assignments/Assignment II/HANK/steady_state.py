# find steady state

import time
import numpy as np
from consav import elapsed
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from scipy.optimize import root

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

    # a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # z
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
            income = (1-par.tau)*ss.w*ss.L*z - ss.Taxes

            c = (1+ss.r)*par.a_grid + income
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]


def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. Normalize labor to 1
    ss.L = 1.0

    # b. Zero inflation steady state 
    ss.pi = 0.0
    
    # c. shock 
    ss.epsT = 0.

    # d. interest rate and assets 
    ss.r = par.r_target_ss
    ss.B = par.B_target_ss
    ss.A = ss.B

    # de. firms
    ss.w = 1/par.mu
    ss.Y = ss.L
    ss.profits = ss.Y - ss.w*ss.L

    # f. government
    ss.Taxes = ss.r*ss.B 
    ss.LT = ss.Taxes - par.tau*(ss.w*ss.L + ss.profits)
    
    # g. households 
    if par.HH_type == 'HANK':

        prepare_hh_ss(model)
        model.solve_hh_ss(do_print=do_print)
        model.simulate_hh_ss(do_print=do_print) 

    elif par.HH_type == 'TANK':
        
        ss.A_hh = ss.A 
        ss.C_HtM = (1-par.tau)*(ss.w*ss.L + ss.profits) - ss.LT
        ss.C_R = (1-par.tau)*(ss.w*ss.L + ss.profits) + ss.r*ss.A_hh/(1-par.sHtM) - ss.LT
        ss.C_hh = ss.C_R*(1-par.sHtM) + ss.C_HtM*par.sHtM
        ss.MUC_hh = ss.C_R**(-par.sigma)*(1-par.sHtM) + ss.C_HtM**(-par.sigma)*par.sHtM
        
    else:

        raise ValueError('HH_type must be "HANK" or "TANK"')

    # h. market clearing
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh 

    # i. labor supply 
    par.varphi = ((1-par.tau)*ss.w*ss.MUC_hh)/ss.L**par.nu
    ss.labor_supply_res = 0.0 # used to derive par.varphi

def obj_ss(x, model, root=True, do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta = x[0]
    par.B_target_ss = x[1]

    evaluate_ss(model,do_print=do_print)

    # MPCs  
    model._compute_jac_hh(inputs_hh_all=['LT'])
    ann_mpcs = aggregate_to_annual(-model.jac_hh[('C_hh','LT')][:,0])

    residual = np.array([ss.clearing_A, ann_mpcs[0] - par.ann_mpc_target]) 

    if root:
        return residual
    else:
        return np.sum(np.abs(residual))

def find_ss(model,do_print=False,x0=None, calibrate=True):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    if calibrate:

        if x0 is None:
            x0 = np.array([0.97354, 2.7245]) # beta and B_target_ss
        
        # root 
        res = root(obj_ss, x0, args=(model,True,do_print), method='hybr', tol=1e-08)
        assert res.success

        # final evaluation
        obj_ss(res.x,model)

    else:

        evaluate_ss(model)

    # b. print
    print(f'steady state found in {elapsed(t0)}')
    print(f' beta = {par.beta:8.4f}')
    print(f' r    = {ss.r:8.4f}')
    print(f' B   = {ss.B:8.4f}')
    print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
    print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
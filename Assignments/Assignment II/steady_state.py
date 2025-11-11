# find steady state

import time
import numpy as np
from consav import elapsed
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from scipy.optimize import root

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
    par.z_grid[:],ss.z_trans[:,:,:],par.e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)
    par.share_low = np.sum(par.e_ergodic[:4]) # share of low income households, needed for targeted transfers
    
    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,0] = par.e_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]
            income = (1-par.tax)*ss.L*z - ss.Taxes

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
    
    # c. interest rate and assets 
    ss.r = par.r
    ss.B = par.B
    ss.A = ss.B

    # d. firms
    ss.Y = ss.L

    # e. government
    par.G = par.tax - ss.r*ss.B
    ss.tax = par.tax
    ss.Taxes = ss.tax * ss.Y
    ss.transfers = par.transfers
    
    # g. households 
    if par.HH_type == 'HANK':

        prepare_hh_ss(model)
        model.solve_hh_ss(do_print=do_print)
        model.simulate_hh_ss(do_print=do_print) 

    elif par.HH_type == 'TANK':
        
        ss.A_hh = ss.A 
        ss.C_HtM = (1-ss.tax)*ss.L + ss.transfers
        ss.C_R = (1-ss.tax)*ss.L + ss.r*ss.A_hh/(1-par.sHtM) + ss.transfers
        ss.C_hh = ss.C_R*(1-par.sHtM) + ss.C_HtM*par.sHtM
        ss.MUC_hh = ss.C_R**(-par.sigma)*(1-par.sHtM) + ss.C_HtM**(-par.sigma)*par.sHtM
        
    elif par.HH_type == 'RANK':

        ss.A_hh = ss.A 
        ss.C_R = (1-ss.tax)*ss.L + ss.r*ss.A_hh + ss.transfers
        ss.C_hh = ss.C_R
        ss.MUC_hh = ss.C_R**(-par.sigma)
    else:

        raise ValueError('HH_type must be "HANK" or "TANK"')

    # h. market clearing
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh-par.G

    # i. labor supply 
    par.varphi = ((1-ss.tax)*ss.MUC_hh)/ss.L**par.nu
    ss.NKPC_res = 0.0 # used to derive par.varphi

def obj_ss(x, model, do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta = x[0]
    
    evaluate_ss(model,do_print=do_print)

    return ss.clearing_A
    
def find_ss(model,do_print=False,x0=None):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    if x0 is None:
        x0 = np.array([0.9735421945492544])
        
    # root 
    res = root(obj_ss, x0, args=(model,do_print), method='hybr', tol=1e-08)
    assert res.success

    # final evaluation
    obj_ss(res.x,model)


    # b. print
    print(f'steady state found in {elapsed(t0)}')
    print(f' beta = {par.beta:8.4f}')
    print(f' r    = {ss.r:8.4f}')
    print(f' B   = {ss.B:8.4f}')
    print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
    print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
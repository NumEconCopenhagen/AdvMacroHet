# find steady state

import time
import numpy as np
from scipy import optimize

from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # a. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # b. productivity
    par.z_grid[:],ss.z_trans[0,:,:],par.z_ergodic[:],_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    ss.Dbeg[0,:,0] = par.z_ergodic
    ss.Dbeg[0,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # low consumption, so households save from the first iteration (the solver stops when the policy for a stops changing)
    c = ss.ra*par.a_grid[None,:] + 0.5*ss.Z*par.z_grid[:,None]
    ss.vbeg_a[0] = ss.z_trans[0]@c**(-par.sigma)

def obj_ss(beta,model,do_print=False):
    """ evaluate the steady state for a given beta and return the bond market error """

    par = model.par
    ss = model.ss
    par.beta = beta

    # a. fixed
    ss.L = ss.Y = 1.0
    ss.pi_w = 0.0
    ss.eps_r = 0.0
    
    # b. government
    ss.B = par.B_target_ss
    ss.r = ss.ra = par.r_target_ss
    ss.taxes = ss.r*ss.B + par.G
    ss.tau = ss.taxes/ss.Y

    # c. income
    ss.Z = (1-ss.tau)*ss.L
    par.Z_ss = ss.Z
    
    # d. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # e. market clearing
    ss.clearing_A = ss.B-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh-par.G

    # f. NK wage curve
    par.varphi = ((1-ss.tau)*ss.C_hh**(-par.sigma))/ss.L**par.nu
    ss.NKWC_res = 0.0 # used to derive par.varphi

    return ss.clearing_A

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    t0 = time.time()

    res = optimize.root_scalar(obj_ss,bracket=(0.5,1/(1+par.r_target_ss)-1e-4),method='brentq',args=(model,),xtol=1e-10)
    obj_ss(res.root,model)

    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' beta  = {par.beta:8.4f}')
        print(f' tau   = {ss.tau:8.4f}')
        print(f' C     = {ss.C_hh:8.4f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')

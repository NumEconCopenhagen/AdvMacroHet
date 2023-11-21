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

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. z
    par.z_grid[:],ss.z_trans[0,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,0] = e_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    model.set_hh_initial_guess() # calls .solve_hh_backwards() with ss=True
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.Z = 1.0
    ss.N = 1.0
    ss.pi = 0.0
    
    # b. targets
    ss.r = par.r_target_ss
    ss.A = ss.B = par.B_target_ss
    ss.G = par.G_target_ss

    # c.. monetary policy
    ss.i = ss.istar = (1+ss.r)*(1+ss.pi)-1

    # d. firms
    ss.Y = ss.Z*ss.N
    ss.w = ss.Z/par.mu
    ss.s = ss.w/ss.Z
    ss.NKPC_res = 0.0

    ss.adjcost = 0.0 # by definition

    ss.d = (1-ss.s)*ss.Y-ss.adjcost
    
    # e. government
    ss.tau = ss.r*ss.B + ss.G

    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # g. market clearing
    ss.clearing_A = ss.A - ss.A_hh
    ss.clearing_N = ss.N - ss.N_hh
    ss.clearing_Y = ss.Y - (ss.C_hh+ss.G+ss.adjcost)

def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta = x[0]
    par.varphi = x[1]

    evaluate_ss(model,do_print=do_print)
    
    return np.array([ss.clearing_A,ss.clearing_N])

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()
    res = optimize.root(objective_ss,[par.beta, par.varphi],method='hybr',tol=par.tol_ss,args=(model))

    # final evaluation
    objective_ss(res.x,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' beta   = {res.x[0]:8.4f}')
        print(f' varphi = {res.x[1]:8.4f}')
        print('')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in N = {ss.clearing_N:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')

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
            income = (1-ss.tau)*ss.w*ss.L*z+ss.chi

            c = (1+ss.ra)*par.a_grid + income
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.Gamma = 1.0
    ss.chi = 0.0
    ss.L = 1.0
    ss.pi = ss.pi_w = 0.0
    
    # b. targets
    ss.G = par.G_target_ss

    # c.. monetary policy
    ss.ra = ss.i = ss.r = par.r_target_ss
    ss.q = 1/(1+ss.r-par.delta)
    ss.B = par.qB_target_ss/ss.q
    ss.A = ss.q*ss.B

    # d. firms
    ss.w = ss.Gamma
    ss.Y = ss.Gamma*ss.L

    # e. government
    ss.tau = ((1+par.delta*ss.q)*ss.B + ss.G + ss.chi - ss.q*ss.B)/ss.Y
    
    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # g. market clearing
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.G

    # h. NK wage curve
    par.varphi = (1/par.mu*(1-ss.tau)*ss.w*ss.C_hh**(-par.sigma))/ss.L**par.nu
    ss.NKWC_res = 0.0 # used to derive par.varphi

def obj_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta = x
    evaluate_ss(model,do_print=do_print)
    
    return ss.clearing_A

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    beta_min = 0.85**(1/4)
    beta_max = 1/(1+par.r_target_ss)-1e-4

    res = optimize.root_scalar(obj_ss,bracket=(beta_min,beta_max),method='brentq',args=(model,))

    # final evaluation
    obj_ss(res.root,model)
    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' beta = {par.beta:8.4f}')
        print(f' r    = {ss.r:8.4f}')
        print(f' q    = {ss.q:8.4f}')
        print(f' qB   = {ss.q*ss.B:8.4f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
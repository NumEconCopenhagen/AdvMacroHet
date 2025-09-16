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

    # b. beta 
    par.beta_grid[:] = np.array([par.beta_low, par.beta_high])

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
    ss.Y = 1. 
    
    # b. monetary policy
    ss.ra = ss.i = ss.r = par.r_target_ss

    # c. firms
    ss.rK = ss.r + par.delta 
    ss.K = ss.Y*par.K_Y_target 
    par.alpha = par.mu*ss.rK * ss.Y *ss.K   
    ss.w = ss.Y/(ss.L * par.mu / (1-par.alpha))
    par.Gamma  = ss.Y / (ss.K ** par.alpha * ss.L ** (1 - par.alpha)) 
    ss.I = ss.K*par.delta 
    ss.Div = ss.Y - ss.w*ss.L - ss.I

    # d. government
    ss.G = par.G_target_ss * model.ss.Y 
    ss.B = par.B_target_ss * model.ss.Y 
    ss.chi = 0. 
    ss.Taxes = ss.r*ss.B + ss.G - ss.chi 
    ss.tau = ss.Taxes/(ss.w*ss.L)
    ss.Z = ss.w*ss.L * (1-ss.tau)
    
    # e. household 
    model.solve_hh_ss(do_print=False)
    model.simulate_hh_ss(do_print=False)

    # f. market clearing
    ss.pD = ss.Div/ss.r 

    ss.clearing_A = ss.A_hh - (ss.B + ss.pD)
    ss.clearing_Y = ss.Y-ss.C_hh-ss.G-ss.I 

    # g. NK wage curve
    par.varphi = ((1-ss.tau)/par.mu*ss.w*ss.C_hh**(-par.sigma))/ss.L**par.nu
    ss.NKWC_res = 0.0 # used to derive par.varphi

    par.beta = np.mean(par.beta_grid)

def obj_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta_low,par.beta_high = x
    evaluate_ss(model,do_print=do_print)

    
    MPC = np.sum(ss.D[:,:,:-1]*(ss.c[:,:,1:]-ss.c[:,:,:-1])/((1+ss.ra)*(par.a_grid[1:]-par.a_grid[:-1])))

    return np.array([ss.clearing_A, MPC - par.MPC_target])

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # find steady state
    x0 = np.array([par.beta_low,par.beta_high])
    sol = optimize.root(obj_ss, x0, method='hybr', args=(model,do_print))
    print(sol)

    # b. print
    if do_print:

        print(f' Y = {ss.Y:8.4f}')
        print(f' r    = {ss.r:8.4f}')
        print(f' A    = {ss.A_hh:8.4f}')
        print(f' pD    = {ss.pD:8.4f}')
        print(f' B   = {ss.B:8.4f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
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
    z_grid_e,z_trans_e,z_ergodic_e,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz_e)

    # c. discount factors
    beta_trans = np.array([[1-par.q+par.q*(1-par.share_H),par.q*par.share_H],
                           [par.q*(1-par.share_H),1-par.q+par.q*par.share_H]])

    beta_ergodic = np.array([1-par.share_H,par.share_H])

    # d. combine
    par.z_grid[:] = np.tile(z_grid_e,par.Nbeta)
    par.beta_grid[:] = np.repeat(np.array([par.beta_L,par.beta_H]),par.Nz_e)
    ss.z_trans[0,:,:] = np.kron(beta_trans,z_trans_e)
    z_ergodic = np.kron(beta_ergodic,z_ergodic_e)

    ###########################
    # 2. initial distribution #
    ###########################
    
    ss.Dbeg[0,:,0] = z_ergodic
    ss.Dbeg[0,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_z in range(par.Nz):

        income = ss.Z*par.z_grid[i_z]
        c = (1+ss.ra)*par.a_grid + income
        v_a[0,i_z,:] = c**(-par.sigma)

    ss.vbeg_a[0] = ss.z_trans[0]@v_a[0]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.L = ss.Y = 1.0
    ss.pi = ss.pi_w = 0.0
    
    # b. targets
    ss.G = par.G_target_ss
    ss.B = par.B_target_ss 
    ss.A = par.A_target_ss
    ss.p = ss.A-ss.B
    
    # c. monetary policy
    ss.r = ss.rb = ss.ra = par.r_target_ss
    ss.i = ss.r
    ss.q = 1/(1+ss.i-par.delta_B)

    # d. government
    ss.taxes = ss.r*ss.B + ss.G
    ss.tau = ss.taxes/ss.Y

    # e. firms: the markup gives the dividend implied by the stock price
    ss.d = ss.r*ss.p
    par.mu = 1/(1-ss.d/((1-ss.tau)*ss.Y))
    ss.w = 1/par.mu
    ss.Z = (1-ss.tau)*ss.w*ss.L
    ss.chi = 0.0
    
    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # g. market clearing
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.G

    # h. NK wage curve
    par.beta_avg = par.share_H*par.beta_H + (1-par.share_H)*par.beta_L
    par.varphi = ((1-ss.tau)*ss.w*ss.C_hh**(-par.sigma))/ss.L**par.nu
    ss.NKWC_res = 0.0 # used to derive par.varphi

def obj_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    model.par.beta_H = x
    if model.par.model_type == 'IKC': model.par.beta_L = x
    evaluate_ss(model,do_print=do_print)
    
    return model.ss.clearing_A

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    t0 = time.time()

    bracket = (0.80,1/(1+par.r_target_ss)-1e-4) if par.model_type == 'IKC' else (0.90,1.01)
    res = optimize.root_scalar(obj_ss,bracket=bracket,method='brentq',args=(model,),xtol=1e-10)
    obj_ss(res.root,model)

    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' beta_L = {par.beta_L:8.4f}, beta_H = {par.beta_H:8.4f}')
        print(f' mu     = {par.mu:8.4f}')
        print(f' tau    = {ss.tau:8.4f}')
        print(f' p      = {ss.p:8.4f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')

import numpy as np
from scipy import optimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ set the grids, the transition matrix and the initial guesses for the household block """

    par = model.par
    ss = model.ss

    # a. grids
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    # b. transition matrix and initial distribution (everybody at a_lag = 0)
    ss.z_trans[0,:,:] = z_trans
    ss.Dbeg[0,:,0] = z_ergodic/par.Nfix
    ss.Dbeg[0,:,1:] = 0.0

    # c. initial guess for the intertemporal variable
    y = ss.w*par.z_grid
    c = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(K_ss,model,do_print=False):
    """ asset market clearing error for a guess on steady state capital """

    par = model.par
    ss = model.ss

    # a. firms
    ss.K = ss.A = K_ss
    ss.L = 1.0
    ss.Gamma = par.Gamma_ss
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)
    ss.rK = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1.0)
    ss.r = ss.rK-par.delta
    ss.w = (1.0-par.alpha)*ss.Gamma*(ss.K/ss.L)**par.alpha

    # b. households: backward step (EGM), then forward step (histogram method)
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # c. market clearing
    ss.I = par.delta*ss.K
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_L = ss.L-1.0
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

    if do_print: print(f'K = {ss.K:8.4f}, r = {ss.r:7.4f}, w = {ss.w:7.4f}, clearing_A = {ss.clearing_A:12.8f}\n')

    return ss.clearing_A

def find_ss(model,K_min,K_max,do_print=False):
    """ find the steady state with a root-finder on capital """

    optimize.brentq(obj_ss,K_min,K_max,args=(model,do_print))

def obj_ss_beta(beta,model,do_print=False):
    """ asset market clearing error for a guess on the discount factor """

    par = model.par
    ss = model.ss

    par.beta = beta

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    ss.clearing_A = ss.A-ss.A_hh

    if do_print: print(f'beta = {par.beta:7.4f}, clearing_A = {ss.clearing_A:12.8f}\n')

    return ss.clearing_A

def calibrate_ss(model,do_print=False):
    """ calibrate Gamma, delta and beta to the targets for Y, r and K """

    par = model.par
    ss = model.ss

    # a. targets
    ss.L = 1.0
    ss.Y = par.Y_ss_target
    ss.K = ss.A = par.K_ss_target
    ss.r = par.r_ss_target

    # b. closed forms: Gamma from Y, delta from r
    ss.Gamma = par.Gamma_ss = ss.Y/(ss.K**par.alpha*ss.L**(1-par.alpha))
    ss.rK = par.alpha*ss.Y/ss.K
    par.delta = ss.rK-ss.r
    ss.w = (1-par.alpha)*ss.Y/ss.L

    # c. bracket beta: raise beta towards 1/(1+r) until households hold more than K
    beta_grid = (1.0-np.array([0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005]))/(1+ss.r)
    for i_beta,beta in enumerate(beta_grid):
        if obj_ss_beta(beta,model) < 0.0: break

    assert i_beta > 0 and ss.clearing_A < 0.0, 'beta bracket not found'

    # d. beta by root-finding: households must hold A = K
    optimize.brentq(obj_ss_beta,beta_grid[i_beta-1],beta_grid[i_beta],args=(model,do_print))

    # e. remaining variables
    ss.I = par.delta*ss.K
    ss.clearing_L = ss.L-1.0
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

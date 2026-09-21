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
    par.beta_grid[:] = np.linspace(par.beta_mean-par.beta_delta,par.beta_mean+par.beta_delta,par.Nfix)

    # b. transition matrix and initial distribution (equal shares of each beta type, everybody at a_lag = 0)
    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dbeg[i_fix,:,0] = z_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,1:] = 0.0

    # c. initial guess for the intertemporal variable
    y = ss.w*par.z_grid
    c = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(beta_mean,model,do_print=False):
    """ asset market clearing error for a guess on the mean discount factor """

    par = model.par
    ss = model.ss

    par.beta_mean = beta_mean

    # a. households: backward step (EGM), then forward step (histogram method)
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # b. market clearing
    ss.clearing_A = ss.A-ss.A_hh

    if do_print: print(f'beta_mean = {par.beta_mean:7.4f}, clearing_A = {ss.clearing_A:12.8f}\n')

    return ss.clearing_A

def find_ss(model,beta_min=0.5,do_print=False):
    """ calibrate alpha, Gamma, delta and the mean discount factor to the targets for the labor share, Y, r and K """

    par = model.par
    ss = model.ss

    # a. targets
    ss.L = ss.L_hh = 1.0
    ss.Y = par.Y_ss_target
    ss.K = ss.A = par.K_ss_target
    ss.r = par.r_ss_target

    # b. parameters with a closed form: alpha from the labor share, Gamma from Y, delta from r
    par.alpha = 1-par.labor_share_ss_target
    ss.Gamma = par.Gamma_ss = ss.Y/(ss.K**par.alpha*ss.L**(1-par.alpha))
    ss.rK = par.alpha*ss.Y/ss.K
    par.delta = ss.rK-ss.r
    ss.w = (1-par.alpha)*ss.Y/ss.L

    # c. upper bound: savings of the most patient type are only finite if beta*(1+r) < 1
    beta_max = 1/(1+ss.r)-par.beta_delta-1e-4

    # d. mean discount factor such that households hold A = K
    optimize.brentq(obj_ss,beta_min,beta_max,args=(model,do_print))

    # e. remaining variables
    ss.I = par.delta*ss.K
    ss.clearing_L = ss.L-ss.L_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

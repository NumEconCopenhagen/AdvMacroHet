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
    ss.L = ss.L_hh = 1.0
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
    ss.clearing_L = ss.L-ss.L_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

    if do_print: print(f'K = {ss.K:8.4f}, r = {ss.r:7.4f}, w = {ss.w:7.4f}, clearing_A = {ss.clearing_A:12.8f}\n')

    return ss.clearing_A

def find_ss(model,K_min,K_max,do_print=False):
    """ find the steady state with a root-finder on capital """

    optimize.brentq(obj_ss,K_min,K_max,args=(model,do_print))

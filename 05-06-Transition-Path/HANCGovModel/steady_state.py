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
    ss.Dbeg[0,:,0] = z_ergodic
    ss.Dbeg[0,:,1:] = 0.0

    # c. initial guess for the intertemporal variable
    y = (1-ss.tau)*par.z_grid
    c = par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = c**(-par.sigma)
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(pB,model,do_print=False):
    """ bond market clearing error for a guess on the steady state bond price """

    par = model.par
    ss = model.ss

    # a. government: debt from the steady state budget constraint
    ss.pB = pB
    ss.B = (ss.tau-ss.G)/(1-ss.pB)

    # b. households: backward step (EGM), then forward step (histogram method)
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # c. market clearing
    ss.clearing_B = ss.B-ss.A_hh
    ss.clearing_Y = 1.0-ss.C_hh-ss.G

    if do_print: print(f'pB = {ss.pB:8.6f}, B = {ss.B:8.4f}, clearing_B = {ss.clearing_B:12.8f}\n')

    return ss.clearing_B

def find_ss(model,tau,do_print=False):
    """ find the steady state with a root-finder on the bond price """

    par = model.par
    ss = model.ss

    # a. government: positive debt requires a primary surplus
    assert tau > par.G_ss
    ss.G = par.G_ss
    ss.tau = tau

    # b. bracket: savings are only finite if beta/pB < 1, and debt is only finite if pB < 1
    pB_min = par.beta+1e-3
    pB_max = 1.0-1e-4

    optimize.brentq(obj_ss,pB_min,pB_max,args=(model,do_print))

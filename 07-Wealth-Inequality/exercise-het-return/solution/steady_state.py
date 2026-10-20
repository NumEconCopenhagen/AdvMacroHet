import numpy as np
from scipy import optimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, rouwenhorst

def prepare_hh_ss(model):
    """ set the grids, the transition matrix and the initial guesses for the household block """

    par = model.par
    ss = model.ss

    # a. grids
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    par.e_grid[:],e_trans,e_ergodic,_,_ = log_rouwenhorst(par.rho_e,par.sigma_psi,par.Ne)
    par.r_grid[:],r_trans,r_ergodic,_,_ = rouwenhorst(par.rX_mean*(1-par.rho_r),par.rho_r,par.sigma_r,par.Nr)

    # b. transition matrix and initial distribution (everybody at a_lag = 0)
    ss.z_trans[0,:,:] = np.kron(r_trans,e_trans)
    ss.Dbeg[0,:,0] = np.kron(r_ergodic,e_ergodic)
    ss.Dbeg[0,:,1:] = 0.0

    # c. initial guess for the intertemporal variable
    c = (1+ss.r)*par.a_grid + ss.w
    ss.vbeg_a[0,:,:] = (1+ss.r)*c**(-par.sigma)

def obj_ss(x,model):
    """ errors in the two calibration targets for a guess on beta and rX_mean """

    par = model.par
    ss = model.ss

    par.beta = x[0]
    par.rX_mean = x[1]

    model.solve_hh_ss()
    model.simulate_hh_ss()

    ss.clearing_A = ss.A-ss.A_hh
    clearing_returns = ss.RETURNS_hh-ss.r*ss.K

    return np.array([ss.clearing_A,clearing_returns])

def find_ss(model,x0=np.array([0.949,-0.004]),do_print=False):
    """ calibrate Gamma, delta, beta and rX_mean to the targets for Y, K/Y and r """

    par = model.par
    ss = model.ss

    # a. targets
    ss.L = 1.0
    ss.Y = 1.0
    ss.K = ss.A = par.KY_ss_target*ss.Y
    ss.r = par.r_ss_target

    # b. closed forms: Gamma from Y, delta from r
    ss.Gamma = ss.Y/(ss.K**par.alpha*ss.L**(1-par.alpha))
    ss.rK = par.alpha*ss.Y/ss.K
    par.delta = ss.rK-ss.r
    ss.w = (1-par.alpha)*ss.Y/ss.L

    # c. beta and rX_mean: households hold A = K and the fund pays out r*K
    result = optimize.root(obj_ss,x0,method='hybr',args=(model,))
    assert result.success
    obj_ss(result.x,model)

    # d. remaining variables
    ss.I = par.delta*ss.K
    ss.clearing_L = ss.L-1.0
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

    if do_print:
        print(f'beta = {par.beta:.4f}, rX_mean = {par.rX_mean:.4f}')
        print(f'clearing_A = {ss.clearing_A:.2e}, RETURNS_hh/A_hh = {ss.RETURNS_hh/ss.A_hh:.4f}, clearing_Y = {ss.clearing_Y:.2e}')

import time
import numpy as np

from EconModel import jit

from consav.grids import equilogspace
from consav.markov import find_ergodic
from consav.misc import elapsed

import household_problem

def set_z_trans_ss(model):
    """ set z_trans """

    ss = model.ss

    s = np.zeros_like(ss.a)
    with jit(model) as model_jit:
        household_problem.fill_s(model_jit.par,s)
        household_problem.fill_z_trans(model_jit.par,ss.z_trans,ss.delta,ss.lambda_u_s,s)

def get_Dz(model):
    """ get distribution for z """

    ss = model.ss

    return find_ergodic(ss.z_trans[0,0])

def set_Dbeg_ss(model):
    """ set initial distribution """

    par = model.par
    ss = model.ss

    Dz = get_Dz(model)
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,0] = par.beta_shares[i_fix]*Dz 
        ss.Dbeg[i_fix,:,1:] = 0.0      

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    # a. beta
    par.beta_grid[0] = par.beta_HtM
    par.beta_grid[1] = par.beta_BS
    par.beta_grid[2] = par.beta_PIH 

    # shares
    par.beta_shares = np.zeros(par.Nfix)
    par.beta_shares[0] = par.HtM_share
    par.beta_shares[1] = 1-par.HtM_share-par.PIH_share
    par.beta_shares[2] = par.PIH_share
    
    # b. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)

    # c. z       
    par.z_grid[:] = np.nan # not used
    
    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    # a. transition matrix
    set_z_trans_ss(model)

    # b. ergodic
    set_Dbeg_ss(model)
    
    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    model.set_hh_initial_guess()

def find_ss(model,do_print=False,fix_RealR=False):
    """ find the steady state"""

    par = model.par
    ss = model.ss

    t0 = time.time()
    
    # a. SAM and HANK
    find_ss_SAM(model,do_print=do_print)
    find_ss_HANK(model,do_print=do_print)

    # c. zero errors
    ss.errors_Vj = 0.0
    ss.errors_Vv = 0.0
    #ss.errors_vt = 0.0
    ss.errors_u = 0.0
    ss.errors_pi = 0.0
    ss.errors_assets = 0.0
    ss.errors_U = 0.0
    ss.errors_U_UI = 0.0

    ss.r_ann = (1+ss.r)**12-1
    ss.i_ann = (1+ss.i)**12-1
    ss.pi_ann = (1+ss.pi)**12-1

    if do_print: print(f'steady state found in {elapsed(t0)}')

def find_ss_SAM(model,do_print=False):
    """ find the steady state - SAM """

    par = model.par
    ss = model.ss
    
    # a. shocks
    ss.TFP = 1.0

    # b. fixed
    ss.delta = par.delta_ss
    ss.lambda_u_s = par.lambda_u_s_ss
    ss.theta = par.theta_ss

    ss.px = (par.epsilon-1)/par.epsilon
    ss.w = par.w_share_ss*ss.px

    # c. direct implications
    par.A = ss.lambda_u_s/ss.theta**(1-par.alpha)
    ss.lambda_v = par.A*ss.theta**(-par.alpha)

    # d. labor market dynamics
    ss.u = ss.delta/(ss.lambda_u_s+ss.delta)
    ss.S = ss.u # because s(u_lag) = 1
    ss.v = ss.S*ss.theta

    # e. job and vacancy Bellmans
    ss.Vj = (ss.px*ss.TFP-ss.w)/(1-par.beta_firm*(1-ss.delta))
    par.kappa = ss.lambda_v*ss.Vj

    # f. dividends
    ss.div = ss.TFP*(1-ss.u) - ss.w*(1-ss.u)

    # print
    if do_print:

        print(f'{par.A = :6.4f}')
        print(f'{par.kappa = :6.4f}')
        print(f'{ss.w = :6.4f}')
        print(f'{ss.delta = :6.4f}')
        print(f'{ss.lambda_u_s = :6.4f}')
        print(f'{ss.lambda_v = :6.4f}')
        print(f'{ss.theta = :6.4f}')
        print(f'{ss.u = :6.4f}')
        print(f'{ss.S = :6.4f}')

def find_ss_HANK(model,do_print=False):
    """ find the steady state - HANK """

    par = model.par
    ss = model.ss

    # a. shocks
    pass

    # b. fixed
    ss.pi = 0.0
    ss.r = par.r_ss
    ss.i = (1+ss.r)*(1+ss.pi)-1
    ss.taut = ss.tau = par.tau_ss
    ss.transfer = -ss.div
    
    # c. households
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # checks
    Dz = np.sum(ss.Dbeg,axis=2)
    u_ss = np.sum((par.i_u_hh > 0)*Dz)
    assert np.isclose(u_ss,ss.u)
    assert np.isclose(ss.U_ALL_hh,ss.u)
    assert ss.U_UI_hh <= ss.u

    # d. government
    ss.U_UI_hh_guess = ss.U_UI_hh

    ss.qB = ss.A_hh
    ss.q = 1/(1+ss.r-par.delta_q)
    ss.B = ss.qB/ss.q
    ss.Phi = par.phi_obar*ss.w*ss.U_UI_hh + par.phi_ubar*ss.w*(ss.u-ss.U_UI_hh)
    ss.taxes = ss.tau*(ss.w*(1-ss.u) + ss.Phi)
    expenses_no_G = ((1+par.delta_q*ss.q)*ss.B - ss.q*ss.B) + ss.Phi + ss.transfer
    ss.G = ss.taxes - expenses_no_G
    ss.X = ss.Phi + ss.G + ss.transfer

    # e. clearing_Y
    ss.Y = ss.TFP*(1-ss.u)
    ss.clearing_Y = ss.Y - (ss.C_hh + ss.G) 

    # f. G shock
    par.jump_G = 0.01*ss.G

    if do_print:

        print(f'{ss.G = :6.4f}')
        print(f'{ss.clearing_Y = :6.4f}')     
        print(f'{par.jump_G = :6.4f}')     
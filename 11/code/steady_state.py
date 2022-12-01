# find steady state

import time
import numpy as np
from root_finding import brentq

from EconModel import jit

from consav.grids import equilogspace
from consav.markov import tauchen, find_ergodic
from consav.misc import elapsed

import household_problem

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    # a. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)

    # b. z
    if par.Ne > 1:

        log_e_grid,_,_,_,_ = tauchen(0,par.rho_e,par.sigma_psi,n=par.Ne)       
        par.e_grid[:] = np.exp(log_e_grid)

    else:

        par.e_grid[:] = 1.0
        
    par.z_grid[:] = np.tile(par.e_grid,par.Nu+1)
    
    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    # a. transition matrix
    model.fill_z_trans_ss()

    # b. ergodic
    Dz_raw = find_ergodic(ss.z_trans[0])
    for i_fix in range(par.Nfix):
        ss.Dz[i_fix,:] = par.beta_shares[i_fix]*Dz_raw
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:]    
        ss.Dbeg[i_fix,:,1:] = 0.0      
    
    # c. impose mean-one for z
    par.z_grid[:] = par.z_grid/np.sum(par.z_grid*ss.Dz)

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    model.set_hh_initial_guess()

def find_ss(model,do_print=False,fix_RealR=False):
    """ find the steady state"""

    t0 = time.time()
    
    find_ss_SAM(model,do_print=do_print)
    if not model.par.only_SAM:
        find_ss_HANK(model,do_print=do_print,fix_RealR=fix_RealR)

    if do_print: print(f'steady state found in {elapsed(t0)}')

def find_ss_SAM(model,do_print=False):
    """ find the steady state - SAM """

    par = model.par
    ss = model.ss

    mean_UE = np.mean(model.data['UE'])/100
    mean_EU = np.mean(model.data['EU'])/100
    
    # a. shocks
    ss.shock_TFP = 1.0
    ss.wage_subsidy = 0.0
    ss.hiring_subsidy = 0.0

    # b. fixed
    ss.theta = par.theta_ss
    ss.w = par.w_ss
    ss.px = (par.epsilon_p-1)/par.epsilon_p
    ss.M = ss.px*ss.shock_TFP-ss.w

    ss.lambda_u = mean_UE
    ss.delta = mean_EU/(1-mean_UE)

    # c. direct implications
    par.A = ss.lambda_u/ss.theta**(1-par.alpha)
    ss.lambda_v = par.A*ss.theta**(-par.alpha)

    # d. labor market dynamics
    ss.u = ss.delta*(1-ss.lambda_u)/(ss.lambda_u+ss.delta*(1-ss.lambda_u))
    ss.ut = ss.u/(1-ss.lambda_u)
    ss.vt = ss.ut*ss.theta
    ss.v = (1-ss.lambda_v)*ss.vt
    ss.entry = ss.vt-(1-ss.delta)*ss.v

    # e. job and vacancy bellmans
    if par.exo_sep:

        par.p = np.nan
        par.Upsilon = np.nan
        ss.mu = 0.0

        ss.Vj = ss.M/(1-par.beta_firm*(1-ss.delta))

    else:

        par.p = par.p_fac*ss.delta

        Vj_Upsilon = (ss.delta/par.p)**(-1/par.psi)

        _nom = par.p*Vj_Upsilon**(-1)
        if np.abs(par.psi-1.0) < 1e-8:
            _nom *= np.log(Vj_Upsilon)
        else:
            _nom *= par.psi/(par.psi-1)*(1-Vj_Upsilon**(1-par.psi))

        _denom = (1-par.p*Vj_Upsilon**(-par.psi))

        mu_Vj = _nom/_denom

        ss.Vj = ss.M/(1+par.beta_firm*mu_Vj-par.beta_firm*(1-ss.delta))
        
        par.Upsilon = ss.Vj/Vj_Upsilon
        ss.mu = mu_Vj*ss.Vj

    if par.free_entry:
    
        par.kappa = ss.lambda_v*ss.Vj
        ss.Vv = 0.0
    
    else:

        _fac = 1-par.beta_firm*(1-ss.lambda_v)*(1-ss.delta)
        
        ss.Vv = par.kappa_0
        par.kappa = ss.lambda_v*ss.Vj - _fac*ss.Vv

def find_ss_HANK(model,do_print=False,fix_RealR=False):
    """ find the steady state - HANK """

    par = model.par
    ss = model.ss

    # a. shocks
    ss.shock_beta = 1.0
    ss.G = 0.0
    ss.public_transfer = 0.0

    # b. fixed
    ss.phi_obar = par.phi_obar_ss
    ss.u_bar = par.u_bar_ss
    ss.qB = par.qB_share_ss*ss.w
    ss.Pi = 1.0
    ss.transfer = par.div_hh*(ss.shock_TFP-ss.w)*(1-ss.u) + ss.public_transfer
    
    # c. preparation
    model.prepare_hh_ss()
    ss.U_UI_hh = np.sum(ss.u_UI[:,:,0]*ss.Dz)

    u_UI = np.fmax(np.fmin(ss.u_bar-(par.i_u_hh-1),1.0),0.0)
    u_UI[:,0] = 0.0
    ss.U_UI_hh_guess = np.sum(u_UI*ss.Dz)
    assert np.isclose(ss.U_UI_hh,ss.U_UI_hh_guess)

    u_ss = np.sum((par.i_u_hh > 0)*ss.Dz)
    assert np.isclose(u_ss,ss.u)
    assert ss.U_UI_hh <= ss.u

    # d. equilibrium  
    ss.UI = ss.phi_obar*ss.w*ss.U_UI_hh + par.phi_ubar*ss.w*(ss.u-ss.U_UI_hh)
    ss.Yt_hh = ss.w*(1-ss.u) + ss.UI

    def asset_market_clearing(R):
        
        # o. set
        ss.RealR_ex_post = ss.RealR = R
        ss.q = 1/(ss.RealR-par.delta_q)
        ss.B = ss.qB/ss.q

        ss.tau = ((1+par.delta_q*ss.q)*ss.B+ss.UI-ss.q*ss.B)/ss.Yt_hh

        # oo. solve + simulate
        model.solve_hh_ss(do_print=False)
        model.simulate_hh_ss(do_print=False)

        # ooo. difference
        ss.A_hh = np.sum(ss.a*ss.D)        
        diff = ss.qB - ss.A_hh
        
        return diff

    if fix_RealR:

        assert not np.isnan(ss.RealR)
        asset_market_clearing(ss.RealR)

    else:

        # v. initial values
        R_max = 1.0/par.beta_grid[par.beta_shares > 0.0].max()
        R_min = R_max - 0.05
        R_guess = (R_min+R_max)/2

        diff = asset_market_clearing(R_guess)
        if do_print: print(f'guess:\n     R = {R_guess:12.8f} -> B-A_hh = {diff:12.8f}')

        # vi. find bracket
        if diff > 0:
            dR = R_max-R_guess
        else:
            dR = R_min-R_guess

        if do_print: print(f'find bracket to search in:')
        fac = 0.95
        it = 0
        max_iter = 50
        R = R_guess
        while True:
        
            oldR = R 
            olddiff = diff
            R = R_guess + dR*(1-fac)
            diff = asset_market_clearing(R)
            
            if do_print: print(f'{it:3d}: R = {R:12.8f} -> B-A_hh = {diff:12.8f}')

            if np.sign(diff)*np.sign(olddiff) < 0: 
                break
            else:
                fac *= 0.50 
                it += 1
                if it > max_iter: raise ValueError('could not find bracket')

        if oldR < R:
            a,b = oldR,R
            fa,fb = olddiff,diff
        else:
            a,b = R,oldR
            fa,fb = diff,olddiff                
        
        # vii. search
        if do_print: print(f'brentq:')
        
        brentq(asset_market_clearing,a,b,fa=fa,fb=fb,xtol=par.tol_R,rtol=par.tol_R,
            do_print=do_print,varname='R',funcname='B-A_hh')

        ss.C_hh = np.sum(ss.c*ss.D)

    # iii. R
    ss.R = ss.RealR*ss.Pi
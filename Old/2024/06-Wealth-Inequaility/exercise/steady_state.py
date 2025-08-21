import time
import numpy as np

from consav.grids import equilogspace, nonlinspace
from consav.markov import log_rouwenhorst, rouwenhorst
from consav.misc import elapsed

# simple root finding
import root_finding
from scipy import optimize
import household_problem

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = nonlinspace(0.0,par.a_max,par.Na, 2.)

    # b. z
    par.e_grid[:],e_trans,e_ergodic,_,_ = log_rouwenhorst(par.rho_e,par.sigma_psi,par.Ne)

    # c. r 
    par.r_grid[:],r_trans,r_ergodic,_,_ = rouwenhorst(par.rX_mean*(1-par.rho_r), par.rho_r,par.sigma_r,par.Nr)


    z_ergodic = e_ergodic

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = e_trans
        ss.Dbeg[i_fix,:,0] = z_ergodic/par.Nfix # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    # note: arbitrary to start all with zero assets
        
    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    model.set_hh_initial_guess() # calls .solve_hh_backwards() with ss=True
    

def obj_ss(x,model,do_print):
    """ objective when solving for steady state capital """
    
    par = model.par
    ss = model.ss

    par.beta = x[0] 
    par.rX_mean = x[1]

    # calibrate output 
    ss.Y = 1.0 
    ss.K = ss.Y * par.KY_ss_target
    ss.L = 1. 
    ss.Gamma = ss.Y / (ss.K**par.alpha*ss.L**(1-par.alpha))


    # a. production
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    

    # b. implied prices
    ss.rK = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1.0)
    ss.r = par.r_ss_target 
    par.delta = ss.rK - ss.r
    
    ss.w = (1.0-par.alpha)*ss.Gamma*(ss.K/ss.L)**par.alpha

    # c. household behavior
    if do_print:

        print(f'guess {ss.K = :.4f}')    
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')
    

    prepare_hh_ss(model)
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    ss.A = ss.A_hh

    # d. market clearing
    ss.I = par.delta*ss.K
    ss.clearing_A = ss.RETURNS_hh - ss.r*ss.K    
    ss.clearing_L = ss.L-ss.L_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

    avg_returns = ss.RETURNS_hh/ss.A_hh - par.r_ss_target # target 


    return np.array([ss.clearing_A,  avg_returns]) # targets to hit

def find_ss(model,do_print=False, x0=None):
    """ find steady state using the direct or indirect method """

    t0 = time.time()
    
    find_ss_direct(model, do_print=do_print, x0=x0)

    if do_print: print(f'found steady state in {elapsed(t0)}')


def find_ss_direct(model,do_print=False, x0=None):
    """ find steady state using direct method """

    if x0 is None:
        x0 = np.array([0.89227870 , -0.0021])

    sol = optimize.root(obj_ss, x0, method='Hybr', args=(model,False))
    print(sol)
    obj_ss(sol.x,model,False)
    assert sol.success 

    # f. print
    if do_print:
        ss = model.ss 
        par = model.par 
        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}') # = 0 by construction
        print(f'Discrepancy in L = {ss.clearing_L:12.8f}') # = 0 by construction
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}') # != 0 due to numerical error 

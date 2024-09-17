import time
import numpy as np

from consav.grids import equilogspace, nonlinspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed
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
    par.a_grid[:] = nonlinspace(par.a_min,par.a_max,par.Na, 2.)

    # b. z
    par.z_grid[:],z_trans,par.z_ergodic[:],_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dbeg[i_fix,:,0] = par.z_ergodic/par.Nfix # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    # note: arbitrary to start all with zero assets
        
    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    ss.vbeg_a[:] = (0.05 + par.a_grid*0.1)**(-par.sigma)
    model.set_hh_initial_guess() # calls .solve_hh_backwards() with ss=True
    
    
def obj_ss(x,model,do_print,calibrate):
    """ objective when solving for steady state capital """
    
    par = model.par
    ss = model.ss

    if calibrate: # If we calibrate the model guess on the following variables
        ## CODE HERE ##
        ...

        # in calibration, set tau_l to calibrated value 
        ss.tau_l = par.tau_l_ss

    else: # If we solve the model guess on the following variables
        ## CODE HERE ##
        ...


    # update tau_a value 
    ss.tau_a = par.tau_a_ss

    # a. production
    ss.Gamma = 1.0 # normalize TFP to 1 
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    

    # b. implied prices
    ss.rK = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1.0)
    ss.r = ss.rK - par.delta
    ss.w = (1.0-par.alpha)*ss.Gamma*(ss.K/ss.L)**par.alpha


    # c. household behavior
    prepare_hh_ss(model)
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    ss.A = ss.A_hh

    # d. market clearing
    ss.I = par.delta*ss.K
    ss.clearing_A = ss.K-ss.A_hh
    ss.clearing_L = ss.L-ss.L_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I
    ss.clearing_G = ss.TAXES_hh - ss.transfer 
    
    # e. calibration targets
    KY_res = par.KY_ss_target - ss.K/ss.Y 
    L_HH_res = ss.L_hh - 1.0

    if calibrate: # if calibrating, return calibration targets
        ## CODE HERE ##
        ...
        
    else: # if solving, return modle residuals 
        
        ## CODE HERE ##
        ...
    

def find_ss(model,do_print=False,calibrate=False, x0=None):
    """ find steady state """

    t0 = time.time()

    find_ss_direct(model, do_print=do_print, calibrate=calibrate, x0=x0)

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,do_print=False,calibrate=False, x0=None):
    """ find steady state """

    # Initial guess for root finder  
    if x0 is None:
        if calibrate:
            x0 = ...
        else:
            x0 = ...

    sol = optimize.root(obj_ss, x0, method='hybr', args=(model,False,calibrate))

    # Final evaluation at root 
    obj_ss(sol.x,model,False,calibrate)
    assert sol.success 

    # f. print
    if do_print:
        ss = model.ss 
        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}') 
        print(f'Discrepancy in L = {ss.clearing_L:12.8f}') 
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')  
        print(f'Discrepancy in G = {ss.clearing_G:12.8f}')  


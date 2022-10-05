import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    
    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    ss.z_trans[0,:,:] = z_trans
    ss.Dz[0,:] = z_ergodic
    ss.Dbeg[0,:,0] = ss.Dz[0,:] # ergodic at a_lag = 0.0
    ss.Dbeg[0,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = (1-ss.tau)*par.z_grid
    c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(r_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    ss.r = r_ss

    # a. household behavior
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # b. government
    ss.B = ss.tau/ss.r

    # c. market clearing
    ss.clearing_A = ss.B

    return ss.clearing_A # target to hit
    
def obj_ss(r_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    ss.r = r_ss

    # a. government
    ss.B = ss.tau/ss.r
                       
    # b. households                       
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    
    # d. market clearing
    ss.clearing_B = ss.A_hh-ss.B

    return ss.clearing_B

def find_ss(model,tau,do_print=False,r_min=1e-8,r_max=0.04,Nr=10):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    par = model.par
    ss = model.ss

    if np.isclose(tau,0.0):

        ss.tau = 0.0
        ss.r = 0.0

        model.solve_hh_ss(do_print=do_print)
        model.simulate_hh_ss(do_print=do_print)

        ss.B = ss.A_hh
        ss.clearing_B = ss.A_hh-ss.B

    else:

        ss.tau = tau

        # a. broad search
        if do_print: print(f'### step 1: broad search ###\n')

        r_ss_vec = np.linspace(r_min,r_max,Nr) # trial values
        clearing_B = np.zeros(r_ss_vec.size) # asset market errors

        for i,r_ss in enumerate(r_ss_vec):
            
            try:
                clearing_B[i] = obj_ss(r_ss,model,do_print=do_print)
            except Exception as e:
                clearing_B[i] = np.nan
                print(f'{e}')
                
            if do_print: print(f'clearing_B = {clearing_B[i]:12.8f}\n')
                
        # b. determine search bracket
        if do_print: print(f'### step 2: determine search bracket ###\n')

        r_min = np.max(r_ss_vec[clearing_B < 0])
        r_max = np.min(r_ss_vec[clearing_B > 0])

        if do_print: print(f'r in [{r_min:12.8f},{r_max:12.8f}]\n')

        # c. search
        if do_print: print(f'### step 3: search ###\n')

        root_finding.brentq(
            obj_ss,r_min,r_max,args=(model,),do_print=do_print,
            varname='r_ss',funcname='A_hh-B'
        )

    if do_print: print(f'found steady state in {elapsed(t0)}')
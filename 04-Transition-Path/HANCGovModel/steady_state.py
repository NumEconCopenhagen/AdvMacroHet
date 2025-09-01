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
    ss.Dbeg[0,:,0] = z_ergodic # ergodic at a_lag = 0.0
    ss.Dbeg[0,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = (1-ss.tau)*par.z_grid
    c = m = par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a
    
def obj_ss(pB,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    # a. government
    ss.pB = pB
    ss.B = (ss.G-ss.tau)/(ss.pB-1)
                       
    # b. households                       
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    
    # c. market clearing
    ss.clearing_B = ss.B-ss.A_hh

    return ss.clearing_B

def find_ss(model,tau,do_print=False,pB_min=0.965,pB_max=0.985,Nr=5):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    par = model.par
    ss = model.ss

    assert tau >= par.G_ss, f'tau = {tau} < par.G_ss = {par.G_ss}'

    # a. government
    ss.G = par.G_ss
    ss.tau = tau

    # b. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    pB_vec = np.linspace(pB_min,pB_max,Nr) # trial values
    clearing_B = np.zeros(pB_vec.size) # asset market errors

    for i,pB in enumerate(pB_vec):
        
        try:
            clearing_B[i] = obj_ss(pB,model,do_print=do_print)
        except Exception as e:
            clearing_B[i] = np.nan
            if do_print: print(f'{e}')
            
        if do_print: print(f'clearing_B = {clearing_B[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    pB_min = np.max(pB_vec[clearing_B < 0])
    pB_max = np.min(pB_vec[clearing_B > 0])

    if do_print: print(f'pB in [{pB_min:12.8f},{pB_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,pB_min,pB_max,args=(model,),do_print=do_print,
        varname='pB',funcname='B-A_hh'
    )

    if do_print: print(f'found steady state in {elapsed(t0)}')
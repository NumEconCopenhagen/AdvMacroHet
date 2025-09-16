import time
import numpy as np
from scipy import optimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

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

    for i_fix in range(par.Nfix):

        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dz[i_fix,:] = z_ergodic*0.25
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:] # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    for i_fix in range(par.Nfix):
        
        
        # a. raw value
        ell = 1.0
        y = ss.wt*ell*par.zeta_grid[i_fix]*par.z_grid
        c = m = (1+ss.rt)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
        v_a = (1+ss.rt)*c**(-par.sigma)

        # b. expectation
        ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a

def obj_ss(KL,model,do_print=False):

    par = model.par
    ss = model.ss

    # a. firms
    ss.rK = par.alpha*par.Gamma*(KL)**(par.alpha-1)
    ss.w = (1.0-par.alpha)*par.Gamma*(KL)**par.alpha

    # b. arbitrage
    ss.r = ss.rB = ss.rK - par.delta

    # c. government
    ss.tau_a = par.tau_a_ss
    ss.tau_ell = par.tau_ell_ss

    # d. households
    ss.G = par.G_ss
    ss.transfer = par.transfer_ss
    ss.wt = (1-ss.tau_ell)*ss.w
    ss.rt = (1-ss.tau_a)*ss.r
    
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # e. market clearing
    ss.B = (ss.tau_a*ss.r*ss.A_hh+ss.tau_ell*ss.w*ss.L_hh-ss.G-np.sum(ss.Dz[:,0])*ss.transfer)/ss.r
    ss.L = ss.L_hh
    ss.K = KL*ss.L
    ss.Y = par.Gamma*ss.K**(par.alpha)*ss.L**(1-par.alpha)

    ss.clearing_A = ss.B + ss.K - ss.A_hh
    ss.clearing_N = ss.L - ss.L_hh
    ss.clearing_Y = ss.Y - (ss.C_hh+ss.G+par.delta*ss.K)

    # f. welfare
    ss.U_hh_0 = np.sum(ss.D[0,:,:]*ss.u[0,:,:])*4
    ss.U_hh_1 = np.sum(ss.D[1,:,:]*ss.u[1,:,:])*4
    ss.U_hh_2 = np.sum(ss.D[2,:,:]*ss.u[2,:,:])*4
    ss.U_hh_3 = np.sum(ss.D[3,:,:]*ss.u[3,:,:])*4

    mean_inc = np.sum(ss.D*ss.inc)
    ss.std_inc = np.sqrt(np.sum(ss.D*(ss.inc-mean_inc)**2))

    mean_a = np.sum(ss.D*ss.a)
    ss.std_a = np.sqrt(np.sum(ss.D*(ss.a-mean_a)**2))

    return ss.clearing_A

def find_ss(model,KL_min=None,KL_max=None,do_print=False):
    """ find the steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss

    if KL_min is None: KL_min = ((1/par.beta+par.delta-1)/(par.alpha*par.Gamma))**(1/(par.alpha-1)) + 1e-2
    if KL_max is None: KL_max = (par.delta/(par.alpha*par.Gamma))**(1/(par.alpha-1))-1e-2

    # a. solve for K and L
    if do_print:
        print(f'seaching in [{KL_min:.4f},{KL_max:.4f}]')

    res = optimize.root_scalar(obj_ss,bracket=(KL_min,KL_max),method='brentq',args=(model,))
    
    # b. final evaluations
    obj_ss(res.root,model)

    # c. show
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.K = :6.3f}')
        print(f'{ss.B = :6.3f}')
        print(f'{ss.A_hh = :6.3f}')
        print(f'{ss.L = :6.3f}')
        print(f'{ss.Y = :6.3f}')
        print(f'{ss.r = :6.3f}')
        print(f'{ss.w = :6.3f}')
        print(f'{ss.clearing_A = :.2e}')
        print(f'{ss.clearing_N = :.2e}')
        print(f'{ss.clearing_Y = :.2e}')
import numpy as np
from scipy import optimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, find_ergodic

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. beta
    par.beta_grid[0] = par.beta_breve  

    # b. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # c. z
    s_grid,s_trans,_,_,_ = log_rouwenhorst(par.rho_s,par.sigma_psi,par.Ns)

    par.s_grid[:] = s_grid
    z_trans = s_trans

    # ergodic
    z_ergodic = find_ergodic(z_trans)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dz[i_fix,:] = z_ergodic
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:]
        ss.Dbeg[i_fix,:,1:] = 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            y = ss.w*par.s_grid[i_z]
            r = ss.rK-par.delta
            m = (1+r)*par.a_grid + y
            c = 0.5*m
            v_a = (1+r)*c**(-par.sigma)

            ss.vbeg_a[i_fix,i_z] = np.sum(z_trans[i_z,:,np.newaxis]*v_a[np.newaxis,:],axis=0)

def find_ss(model,do_print=False,r_min=0.00,r_max=0.04,Nr=10):
    """ find steady state using direct method """

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    r_ss_vec = np.linspace(r_min,r_max,Nr) # trial values
    clearing_A = np.zeros(r_ss_vec.size) # asset market errors

    for i,r_ss in enumerate(r_ss_vec):
        
        try:
            obj_ss(r_ss,model,do_print=do_print)
            clearing_A[i] = model.ss.clearing_A
        except Exception as e:
            clearing_A[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_A = {clearing_A[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    r_max = np.min(r_ss_vec[clearing_A > 0])
    r_min = np.max(r_ss_vec[clearing_A < 0])

    if do_print: print(f'r in [{r_min:12.8f},{r_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    res = optimize.root_scalar(obj_ss,bracket=(r_min,r_max),method='brentq',args=(model,))

    if do_print: print(f'done\n')

    obj_ss(res.root,model,do_print=do_print)

def obj_ss(r_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    # a. set real interest rate and rental rate
    ss.r = r_ss
    ss.rK = ss.r+par.delta
    
    # b. production
    ss.Gamma = 1.0
    ss.alpha = par.alpha_ss
    ss.K = (ss.rK/ss.alpha*ss.Gamma)**(1/(ss.alpha-1))
    ss.L = 1.0 # by assumption
    ss.Y = ss.Gamma*ss.K**ss.alpha*ss.L**(1-ss.alpha)    

    # b. implied wage
    ss.w = (1.0-ss.alpha)*ss.Gamma*(ss.K/ss.L)**ss.alpha

    # c. household behavior
    if do_print:

        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    if do_print: print(f'implied {ss.C_hh = :.4f}')
    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. market clearing
    ss.A = ss.K
    ss.clearing_A = ss.A_hh-ss.A

    ss.I = ss.K - (1-par.delta)*ss.K
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

    return ss.clearing_A
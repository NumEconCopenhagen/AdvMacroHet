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

    # CHANGE
    par.beta_grid[0] = par.beta_breve - par.sigma_beta 
    par.beta_grid[1] = par.beta_breve  
    par.beta_grid[2] = par.beta_breve + par.sigma_beta
    # END

    # b. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # c. z
    s_grid,s_trans,_,_,_ = log_rouwenhorst(par.rho_s,par.sigma_psi,par.Ns)

    # REPLACE     

    # par.s_grid[:] = s_grid
    # z_trans = s_trans

    # grids
    par.s_grid[:par.Ns] = s_grid
    par.s_grid[par.Ns:] = s_grid

    par.chi_grid[:par.Ns] = 1-par.sigma_chi*par.pi_chi_obar/par.pi_chi_ubar
    par.chi_grid[par.Ns:] = 1+par.sigma_chi

    # transition
    z_trans = np.zeros((par.Nz,par.Nz))

    z_trans[:par.Ns,:par.Ns] = (1-par.pi_chi_obar)*s_trans
    z_trans[:par.Ns,par.Ns:] = par.pi_chi_obar*s_trans

    z_trans[par.Ns:,:par.Ns] = par.pi_chi_ubar*s_trans
    z_trans[par.Ns:,par.Ns:] = (1-par.pi_chi_ubar)*s_trans

    # END 

    # ergodic
    z_ergodic = find_ergodic(z_trans)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dz[i_fix,:] = z_ergodic/par.Nfix # CHANGE
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:]
        ss.Dbeg[i_fix,:,1:] = 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            y = ss.w*par.s_grid[i_z]
            
            # r = ss.rK-par.delta
            # DELETE
            chi = par.chi_grid[i_z]
            r = ss.rK*chi-par.delta
            # END

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

    # DELTE
    if do_print:    
        ss = model.ss
        print(f'{ss.r = :.4f}')
        print(f'{ss.clearing_A = :12.8f}')
        print(f'{ss.clearing_Y = :12.8f}')        
    # END

    obj_ss(res.root,model,do_print=do_print)

def obj_ss(r_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    ss.tax = 0.0 # DELETE
    ss.transfer = 0.0 # DELETE

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

    # DELETE
    assert np.isclose(1.0,np.sum(par.chi_grid[np.newaxis,:,np.newaxis]*ss.D))
    ss.capital_income = (1-ss.tax)*(ss.rK-par.delta)*ss.K
    ss.policy_target = 0.0
    ss.std_y = np.sqrt(np.sum(ss.Dz*(par.s_grid-1.0)**2))
    ss.std_a = np.sqrt(np.sum(ss.D*(ss.a-ss.A_hh)**2))
    ss.skew_a = np.sum(ss.D*((ss.a-ss.A_hh)/ss.std_a)**3)
    # END

    return ss.clearing_A
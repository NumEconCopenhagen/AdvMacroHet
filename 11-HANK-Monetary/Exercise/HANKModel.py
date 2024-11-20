
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass
import numba as nb 
import household_problem
import steady_state
import blocks

class HANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['Z','ra','chi'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['eps_i'] # exogenous inputs
        self.unknowns = ['pi_w','L'] # endogenous inputs
        self.targets = ['NKWC_res','clearing_A'] # targets
        
        # d. all variables
        self.blocks = [
            'blocks.central_bank',
            'blocks.production',
            'blocks.mutual_fund',
            'blocks.government',
            'hh',
            'blocks.NKWC',
            'blocks.market_clearing'
        ]        

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        
    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. preferences and HH_type 
        par.Nfix = 1
        par.beta = 0.987 # discount factor (determined in ss)
        par.varphi = np.nan # disutility of labor (determined in ss)
        par.HH_type = 'HANK' # 'HANK' or 'RANK'

        par.sigma = 1.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity
        
        # c. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of psi
        par.Nz = 7 # number of productivity states

        # d. price setting
        par.kappa = 0.1 # slope of wage Phillips curve
        par.mu = np.nan # mark-up

        # Government 
        par.do_B = False # bond supply
        par.omega = 0.05 


        # e. firms
        par.Gamma = np.nan # TFP 

        # e. CB 
        par.phi_pi = 1.25 # Taylor rule coefficient on inflation
        par.r_target_ss = 0.02/4 # real interest rate
        

        # f. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 150.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. shocks
        par.jump_eps_i = -0.005 # initial jump
        par.rho_eps_i = 0.80 # AR(1) coefficeint
        par.std_eps_i = 0.00 # std.


        # h. misc.
        par.T = 300 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        par.py_hh = True 
        par.py_blocks = True 

    def allocate(self):
        """ allocate model """

        par = self.par
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss      

    def calc_MPC(self):
        """ MPC """
        
        par = self.par
        ss = self.ss

        MPC = np.sum(ss.D[:,:,:-1]*(ss.c[:,:,1:]-ss.c[:,:,:-1])/((1+ss.ra)*(par.a_grid[1:]-par.a_grid[:-1])))
        iMPC = -self.jac_hh[('C_hh','chi')]
        annual_MPC = np.sum(iMPC[:4,0])
        print(f'{MPC = :.2f}, {iMPC[0,0] = :.2f}')  
        print(f'{annual_MPC = :.2f}')  


    def get_RA_J(self):
            par,ss = self.par, self.ss
            T = par.T 
            
            M_RA = {'C_hh' : {'Z' : np.zeros((T,T)), 'ra' : np.zeros((T,T))}, 
                    'A_hh' : {'Z' : np.zeros((T,T)), 'ra' : np.zeros((T,T))},
                    }

            # linearize RA 
            par,ss = self.par, self.ss
            h = 1e-04 
            Z = np.zeros(T)+self.ss.Z 
            ra = np.zeros(T)+self.ss.ra 
            for s in range(T): # shock at time s 
                    # Z shock 
                    Z_ = Z.copy()
                    Z_[s] += h 
                    C,A = RA_block(ra,Z_, par.sigma, ss.ra, ss.C_hh, ss.A_hh, T)
                    M_RA['C_hh']['Z'][:,s] =  (C - self.ss.C_hh)/h
                    M_RA['A_hh']['Z'][:,s] =  (A - self.ss.A_hh)/h

                    # ra shock
                    ra_ = ra.copy()
                    ra_[s] += h 
                    C,A = RA_block(ra_,Z, par.sigma, ss.ra, ss.C_hh, ss.A_hh, T)
                    M_RA['C_hh']['ra'][:,s] =  (C - self.ss.C_hh)/h
                    M_RA['A_hh']['ra'][:,s] =  (A - self.ss.A_hh)/h

            return M_RA


@nb.njit 
def RA_block(ra, Z, sigma, ss_ra, ss_C, ss_A, T):
    C_hh = np.zeros(T) 
    A_hh = np.zeros(T)

    # euler 
    beta_RA = 1/(1+ss_ra)
    for s in range(T):
        t = T - 1 - s

        if s==0:
            C_p = ss_C
            ra_p = ss_ra 
        else:
            C_p = C_hh[t+1]
            ra_p = ra[t+1]

        C_hh[t] = ((1+ra_p) * beta_RA * C_p**(-sigma))**(-1/sigma)

    # budget
    for t in range(T):
        if t ==0:
            A_lag = ss_A
        else:
            A_lag = A_hh[t-1] 
        
        A_hh[t] = (1+ra[t])*A_lag + Z[t]  - C_hh[t]

    return C_hh, A_hh 

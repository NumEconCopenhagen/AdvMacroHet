
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

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
        self.inputs_hh = ['w','ra','L','tau','chi'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['G','chi','Gamma'] # exogenous inputs
        self.unknowns = ['pi_w','L'] # endogenous inputs
        self.targets = ['NKWC_res','clearing_A'] # targets
        
        # d. all variables
        self.blocks = [
            'blocks.production',
            'blocks.central_bank',
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

        # a. preferences
        par.Nfix = 1
        par.beta = np.nan # discount factor (determined in ss)
        par.varphi = np.nan # disutility of labor (determined in ss)

        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity
        
        # c. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of psi
        par.Nz = 7 # number of productivity states

        # d. price setting
        par.kappa = 0.1 # slope of wage Phillips curve
        par.mu = 1.2 # mark-up

        # e. government
        par.phi_pi = 1.5 # Taylor rule coefficient on inflation
        par.rho_i = 0.90 # Taylor rule persistence
        
        par.G_target_ss = 0.20 # government spending
        par.qB_target_ss = 1.00 # bond supply
        par.r_target_ss = 1.02**(1/4)-1 # real interest rate

        par.omega = 0.10 # tax aggressiveness
        par.delta = 0.80 # persistence of government bonds

        # f. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 50.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. shocks
        par.jump_G = 0.00 # initial jump
        par.rho_G = 0.00 # AR(1) coefficeint
        par.std_G = 0.00 # std.

        # h. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
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
        iMPC = self.jac_hh[('C_hh','chi')]
        print(f'{MPC = :.2f}, {iMPC[0,0] = :.2f}')  

    def calc_fiscal_multiplier(self):
        """ fiscal multiplier """

        par = self.par
        ss = self.ss
        path = self.path

        nom = np.sum([(1+ss.r)**(-t)*(path.Y[t]-ss.Y) for t in range(par.T)])        
        denom = np.sum([(1+ss.r)**(-t)*(path.G[t]-ss.G) for t in range(par.T)])   

        fiscal_multiplier = nom/denom
        print(f'{fiscal_multiplier = :.2f}')

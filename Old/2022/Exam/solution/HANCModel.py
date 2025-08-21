import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem
import blocks

class HANCModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','sim','ss','path']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['rK','w','tax','transfer'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','u','v','rK_a'] # outputs
        self.intertemps_hh = ['vbeg','vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['alpha'] # exogenous shocks
        self.unknowns = ['K',] # endogenous unknowns
        self.targets = ['clearing_A'] # targets = 0

        # d. all variables
        self.varlist = [
            'A','alpha','clearing_A','clearing_Y',
            'Gamma','I','K','L','r','rK','w','Y',
            'std_a','skew_a','std_y','capital_income','tax','transfer','policy_target',
            ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # CHANGE
        par.Nfix = 3 # number of fixed discrete states
        par.Ns = 5 # number of stochastic income states
        par.Nz = 2*5 # number of stochastic discrete states
        # END

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta_breve = 0.96 # discount factor, mean
        par.sigma_beta = 0.01 # discount factor, spread
        par.kappa = 0.5 # weight on utility of wealth
        par.a_ubar = 5.0 # luxury of utility of wealth

        # b. income process
        par.rho_s = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of persistent shock

        # c. return process
        par.sigma_chi = 0.10 # depreciation rate, spread
        par.pi_chi_obar = 0.01 # depreciation rate, spread
        par.pi_chi_ubar = 0.10 # depreciation rate, spread

        # d. production and investment
        par.alpha_ss = 0.30 # cobb-douglas
        par.delta = 0.10 # depreciation rate, mean in ss
        
        # d. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 100 # number of grid points

        # e. shocks
        par.jump_Gamma = 0.00 # initial jump
        par.rho_Gamma = 0.0 # AR(1) coefficient

        par.jump_alpha = 0.01 # initial jump
        par.rho_alpha = 0.90 # AR(1) coefficient

        # . misc.
        par.T = 500 # length of transition path        
        par.simT = 2_000 # length of simulation 
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-12 # tolerance when solving eq. system
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.beta_grid = np.zeros(par.Nfix)
        par.s_grid = np.zeros(par.Nz)
        par.chi_grid = np.zeros(par.Nz) # DELETE
        
        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
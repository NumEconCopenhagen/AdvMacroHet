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
        self.inputs_hh = ['r','w'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c', 'returns'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['Gamma'] # exogenous shocks
        self.unknowns = ['K','L'] # endogenous unknowns
        self.targets = ['clearing_A','clearing_L'] # targets = 0
        self.blocks = [ # list of strings to block-functions
            'blocks.production_firm',
            'blocks.mutual_fund',
            'hh', # household block
            'blocks.market_clearing']

        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1 # number of fixed states
        par.Nr = 7 # number points in r grid
        par.Ne = 5 # number of productivity states 
        par.Nz = par.Ne*par.Nr  

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor, mean, range is [mean-width,mean+width]
        
        # b. income and r parameters
        par.rho_e = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_e**2.0)**0.5 # std. of persistent shock
        par.rho_r = 0.8 # persistence of r process
        par.rX_mean = 0.01 # mean of r process
        par.sigma_r = 0.016 # standard deviation of r process

        # c. production and investment
        par.Gamma_ss = np.nan # technology level [determined in ss]
        par.alpha = 0.36 # cobb-douglas
        par.delta = 0.075 # depreciation [determined in ss]

        # d. calibration
        par.r_ss_target = 0.04 # target for real interest rate
        par.w_ss_target = 1.0 # target for real wage
        par.KY_ss_target = 3.

        # f. grids         
        par.a_max = 500_000. #3000.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. shocks
        par.jump_Gamma = -0.10 # initial jump
        par.rho_Gamma = 0.90 # AR(1) coefficient
        par.std_Gamma = 0.01 # std. of innovation

        # i. misc.
        par.T = 500 # length of transition path        
        par.simT = 2_000 # length of simulation 
        
        par.max_iter_solve = 80_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-10 # tolerance when solving household problem
        par.tol_simulate = 1e-10 # tolerance when simulating household problem
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        par.py_hh = True  # call solve_hh_backwards in Python-model
        par.py_block = True # call blocks in Python-model
        par.full_z_trans = False # let z_trans vary over endogenous states
        par.warnings = True # print warnings if nans are encountered

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.beta_grid = np.zeros(par.Nfix)
        par.e_grid = np.zeros(par.Ne)
        par.r_grid = np.zeros(par.Nr)
        
        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    # remember in model = EconModelClass(name='') we call:
    # self.settings()
    # self.setup()
    # self.allocate()

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] # not used today: 'ini', 'path', 'sim'

        # settings required for in GEModelClass
        # important for allocate_GE in self.allocate()

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','w'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used today)
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = [] # exogenous shocks (not used today)
        self.unknowns = [] # endogenous unknowns (not used today)
        self.targets = [] # targets = 0 (not used today)
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

        par.Nfix = 3 # number of fixed discrete states (here betas)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.sigma = 1.0 # CRRA coefficient
        par.beta_mean = 0.95 # mean discount factor, calibrated in find_ss
        par.beta_delta = 0.1 # types are beta_mean-beta_delta, beta_mean, beta_mean+beta_delta
         
        # b. income parameters
        par.rho_z = 0.9 # AR(1) parameter
        par.sigma_psi = 0.5 # std. of persistent shock

        # c. production and investment
        par.alpha = np.nan # cobb-douglas, backed out from the targets in find_ss
        par.delta = np.nan # depreciation rate, backed out from the targets in find_ss
        par.Gamma_ss = np.nan # technology level in steady state, backed out from the targets in find_ss

        # d. grids
        par.a_max = 10_000.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # e. calibration targets (quarterly)
        par.r_ss_target = 0.05 / 4.0 # 5% annual interest rate
        par.K_ss_target = 4.0 * 4.0  # annual wealth-to-output ratio of 4
        par.Y_ss_target = 1.0 # normalization
        par.labor_share_ss_target = 2/3

        # f. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 100_000 # maximum number of iterations when simulating household problem
        
        par.tol_solve = 1e-10 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. extra grids must be allocated by hand, here one discount factor per fixed type
        par.beta_grid = np.zeros(par.Nfix)

        # b. everything else
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss


import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    # remember in model = EconModelClass(name='') we call:
    # self.settings()ba
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
        self.outputs_hh = ['a','c','l'] # outputs
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

        par.Nfix = 3 # number of fixed discrete states (none here)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta_mean = 0.975 # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = 0.010 # discount factor, width, range is [mean-width,mean+width]

        # b. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock

        # c. production and investment
        par.alpha = 0.36 # cobb-douglas
        par.delta = 0.10 # depreciation rate
        par.Gamma_ss = 1.0 # direct approach: technology level in steady state

        # f. grids         
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.01
        par.w_ss_target = 1.0

        # h. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.Nbeta = par.Nfix
        par.beta_grid = np.zeros(par.Nbeta)

        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    # simular to:
    # def find_ss(self):
    #    steady_state.find_ss(self)
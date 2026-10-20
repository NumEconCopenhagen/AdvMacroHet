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
        self.namespaces = ['par','ini','ss','path','sim']

        # settings required for in GEModelClass
        # important for allocate_GE in self.allocate()

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','w'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c'] # outputs
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

        par.Nfix = 1 # number of fixed discrete states (here none)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta = 0.96 # discount factor [calibrated]
         
        # b. income parameters
        par.rho_z = 0.9 # AR(1) parameter
        par.sigma_psi = 0.2*(1.0-par.rho_z**2)**0.5 # std. of shock, std. of log z is 0.2

        # c. production and investment
        par.alpha = 1/3 # cobb-douglas
        par.delta = 0.08 # depreciation rate [calibrated]
        par.Gamma_ss = 1.0 # technology level in steady state [calibrated]

        # d. grids         
        par.a_max = 10_000.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # e. targets for the calibration exercises
        par.r_ss_target = 0.03 # annual
        par.K_ss_target = 3.0 # K/Y = 3 since Y = 1
        par.Y_ss_target = 1.0

        # f. misc.
        par.T = 500 # length of transition path

        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 100_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iterations when solving eq. system

        par.tol_solve = 1e-10 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
    def allocate(self):
        """ allocate model """

        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
    calibrate_ss = steady_state.calibrate_ss


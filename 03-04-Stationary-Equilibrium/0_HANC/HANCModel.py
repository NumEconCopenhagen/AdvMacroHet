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

        par.Nfix = 1 # number of fixed discrete states (here betas)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.sigma = 1.0 # CRRA coefficient
        par.beta = 0.9810226868872732 # discount factor, mean, range is [mean-width,mean+width]
         
        # b. income parameters
        par.rho_z = 0.9 # AR(1) parameter
        par.sigma_psi = 0.5 # std. of persistent shock

        # c. production and investment
        par.alpha = 1/3 # cobb-douglas
        par.delta = 0.008333333333333331 # depreciation rate
        par.Gamma_ss = 0.39685026299204984 # direct approach: technology level in steady state

        # f. grids         
        par.a_max = 10_000.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.05 / 4.0 # quaterly targets
        par.K_ss_target = 4.0 * 4.0  # quarterly targets
        par.Y_ss_target = 1.0

        # h. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 100_000 # maximum number of iterations when simulating household problem
        
        par.tol_solve = 1e-10 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss


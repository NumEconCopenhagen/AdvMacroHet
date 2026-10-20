import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ini','ss','path','sim']

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','w'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','returns'] # outputs
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
        par.Ne = 5 # number of productivity states
        par.Nr = 7 # number of excess return states
        par.Nz = par.Ne*par.Nr # number of stochastic discrete states

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor [calibrated]

        # b. productivity
        par.rho_e = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_e**2.0)**0.5 # std. of shock, std. of log e is 0.30

        # c. excess returns
        par.rho_r = 0.8 # AR(1) parameter
        par.sigma_r = 0.019 # std. of shock
        par.rX_mean = np.nan # mean excess return [calibrated]

        # d. production
        par.alpha = 0.36 # cobb-douglas
        par.delta = np.nan # depreciation rate [calibrated]

        # e. targets for the calibration
        par.r_ss_target = 0.04
        par.KY_ss_target = 3.0

        # f. grids
        par.a_max = 500_000.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. misc.
        par.T = 500 # length of transition path

        par.max_iter_solve = 80_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem

        par.tol_solve = 1e-8 # tolerance when solving household problem (absolute, 1e-10 fails with a_max = 500,000)
        par.tol_simulate = 1e-10 # tolerance when simulating household problem

    def allocate(self):
        """ allocate model """

        par = self.par

        par.e_grid = np.zeros(par.Ne)
        par.r_grid = np.zeros(par.Nr)

        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

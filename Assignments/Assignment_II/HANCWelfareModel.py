import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCWelfareModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] 
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','wt'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used today)
        self.outputs_hh = ['a','c','ell','l','inc','u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = [] # exogenous shocks
        self.unknowns = ['K','L'] # endogenous unknowns
        self.targets = ['clearing_A','clearing_L'] # targets = 0
        self.blocks = [ # list of strings to block-functions
            'blocks.production_firm',
            'blocks.mutual_fund',
            'blocks.government',
            'hh', # household block
            'blocks.market_clearing']

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 4 # number of fixed discrete states (preference and abilities types)
        par.Nz = 7 # number of stochastic discrete states

        # a. preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient
        par.varphi = 1.0 # dis-utility of labor
        par.nu = 1.0 # inverse Frisch elasticity of labor supply

        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.15 # std. of shock

        # c. production and investment
        par.Gamma_Y = 1.0 # technology level
        par.alpha = 0.30 # cobb-douglas coefficient
        par.delta = 0.10 # depreciation rate

        # d. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 100 # number of grid points

        # e. government
        par.tau_ss = 0.00 # tax rate on wage income

        # e. misc.
        par.max_iter_ell = 200 # maximum number of iterations when solving for ell 
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_ell = 1e-12 # tolerance when solving for ell 
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

    def allocate(self):
        """ allocate model """

        par = self.par

        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
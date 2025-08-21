import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] # not used today: 'ini', 'path', 'sim'

        # not used today: .sim and .path
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['rt','wt','transfer'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used today)
        self.outputs_hh = ['a','c','ell','l','inc','u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = [] # exogenous shocks (not used today)
        self.unknowns = [] # endogenous unknowns (not used today)
        self.targets = [] # targets = 0 (not used today)

        # d. all variables
        self.varlist = [
            'Y','K','L','rK','w','r',
            'rB','tau_a','tau_ell','G','B',
            'rt','wt','transfer',
            'clearing_A','clearing_N','clearing_Y',
            'U_hh_0','U_hh_1','U_hh_2','U_hh_3','std_inc','std_a']

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = None # not used today
        self.block_post = None # not used today

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 4 # number of fixed discrete states (preference and abilities types)
        par.Nz = 7 # number of stochastic discrete states

        # a. preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient
        par.varphi_grid = np.array([0.9,0.9,1.1,1.1]) # dis-utility of labor
        par.nu = 1.0 # inverse Frisch elasticity of labor supply

        # b. income parameters
        par.zeta_grid = np.array([0.9,1.1,0.9,1.1])
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.15 # std. of shock

        # c. production and investment
        par.Gamma = 1.0 # technology level
        par.alpha = 0.30 # cobb-douglas coefficient
        par.delta = 0.10 # depreciation rate

        # d. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 100 # number of grid points

        # e. government
        par.G_ss = 0.3 # government spending
        par.tau_a_ss = 0.10 # tax rate on interest-rate income
        par.tau_ell_ss = 0.30 # tax rate on wage income
        par.transfer_ss = 0.00 # tax rate on wage income

        # e. misc.
        par.max_iter_ell = 100 # maximum number of iterations when solving for ell 
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_ell = 1e-10 # tolerance when solving for ell 
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        
    def allocate(self):
        """ allocate model """

        par = self.par

        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
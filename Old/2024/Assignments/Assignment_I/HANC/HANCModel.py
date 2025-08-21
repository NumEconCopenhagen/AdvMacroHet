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
        self.inputs_hh = ['r','w', 'tau_l','tau_a', 'transfer'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','ell', 'l', 'taxes'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = [] # exogenous shocks
        self.unknowns = [] # endogenous unknowns
        self.targets = [] # targets = 0
        self.blocks = [ # list of strings to block-functions
            'blocks.production_firm',
            'blocks.mutual_fund',
            'blocks.transfers',
            'hh', # household block
            'blocks.market_clearing']

        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1 
        par.Nz = 5 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.frisch = 1.0 # CRRA coefficient
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor 
        par.vphi = np.nan # disutility of labor supply 

        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.257 # std. of persistent shock

        # c. production and investment
        par.alpha = 0.33 # cobb-douglas
        par.delta = 0.06 # depreciation 

        # d. calibration
        par.KY_ss_target = 3. # Target K/Y ratio in ss 

        # f. grids         
        par.a_min = 0. 
        par.a_max = 1000. 
        par.Na = 300 # number of grid points

        # g. shocks
        par.jump_Gamma = 0.10 # initial jump
        par.rho_Gamma = 0.90 # AR(1) coefficient
        par.std_Gamma = 0.01 # std. of innovation

        # h. government 
        par.tau_l_ss = 0.3 # tax on labor income 
        par.tau_a_ss = 0.1 # tax on capital income 

        # i. misc.
        par.T = 1000 # length of transition path        
        par.simT = 2_000 # length of simulation 
        
        par.max_iter_solve = 80_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-10 # tolerance when solving household problem
        par.tol_simulate = 1e-10 # tolerance when simulating household problem
        par.tol_broyden = 1e-08 # tolerance when solving eq. system
        
        par.py_hh = True  # call solve_hh_backwards in Python-model
        par.py_block = True # call blocks in Python-model
        par.full_z_trans = False # let z_trans vary over endogenous states
        par.warnings = True # print warnings if nans are encountered

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.z_ergodic = np.zeros(par.Nz)

        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
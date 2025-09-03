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
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['Gamma'] # exogenous shocks
        self.unknowns = ['K','L'] # endogenous unknowns
        self.targets = ['clearing_A','clearing_L'] # targets = 0

        # DAG ordering here 
        self.blocks = [ # list of strings to block-functions
            'blocks.production_firm',
            'blocks.mutual_fund',
            'hh', # household block
            'blocks.market_clearing']

        # Example - not working 
        # self.blocks = [ # list of strings to block-functions
        #     'blocks.mutual_fund',
        #     'blocks.production_firm',
        #     'hh', # household block
        #     'blocks.market_clearing']

        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nbeta = 1 # number of patience types
        par.Nphi = 1 # number of ability types
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.sigma = 1.0 # CRRA coefficient
        par.beta = 0.9498866874132476 # discount factor, mean, range is [mean-width,mean+width]

        # b. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock

        # c. production and investment
        par.alpha = 1/3 # cobb-douglas
        par.Gamma_ss = 1.0/(4.0**par.alpha) # technology level [determined in ss]
        par.delta = 0.1/3 # depreciation

        # f. grids         
        par.a_max = 10_000.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. shocks
        par.jump_Gamma = 0.02 # initial jump
        par.rho_Gamma = 0.90 # AR(1) coefficient
        par.std_Gamma = 0.01 # std. of innovation

        # h. misc.
        par.T = 500 # length of transition path        
        par.simT = 2_000 # length of simulation 
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        par.py_hh = False # call solve_hh_backwards in Python-model
        par.py_block = True # call blocks in Python-model
        par.full_z_trans = False # let z_trans vary over endogenous states
        par.warnings = True # print warnings if nans are encountered

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.Nfix = par.Nbeta*par.Nphi
        par.beta_grid = np.zeros(par.Nfix)
        par.phi_grid = np.zeros(par.Nfix)
        
        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
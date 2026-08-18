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
        self.inputs_hh = ['r','w','upsilon','z_scale'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','z','u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['Gamma', 'upsilon'] # exogenous shocks
        self.unknowns = ['K', 'z_scale'] # endogenous unknowns
        self.targets = ['clearing_A', 'clearing_Z'] # targets = 0
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

        par.Nz = 11 # number of stochastic discrete states (here productivity)
        par.Nfix = 1

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor, mean, range is [mean-width,mean+width]

        # b. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock
        par.upsilon_ss = 1.0 
        
        # c. production and investment
        par.Gamma_ss = np.nan # technology level [determined in ss]
        par.alpha = 1/3 # cobb-douglas
        par.delta = np.nan # depreciation [determined in ss]

        # d. calibration
        par.r_ss_target = 0.05 # target for real interest rate
        par.w_ss_target = 2/3 # target for real wage
        par.Y_ss_target = 1.0 
        par.K_ss_target = 4.0 # W/Y = 4 

        # f. grids         
        par.a_max = 10_000.0 # maximum point in grid for a
        par.Na = 1000 # number of grid points

        # g. shocks
        par.jump_Gamma = 0.01 # initial jump
        par.rho_Gamma = 0.90**4 # AR(1) coefficient
        par.std_Gamma = 0.01 # std. of innovation

        # h. misc.
        par.T = 1000 # length of transition path        
        par.simT = 2_000 # length of simulation 
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-10 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        par.py_hh = False # call solve_hh_backwards in Python-model
        par.py_blocks = True # call blocks in Python-model
        par.full_z_trans = False # let z_trans vary over endogenous states
        par.warnings = True # print warnings if nans are encountered

    def allocate(self):
        """ allocate model """
        
        par = self.par
        par.z_ergodic = np.zeros(par.Nz)
        par.z_trans_ss = np.zeros((par.Nfix,par.Nz,par.Nz))
        par.z_log_grid = np.zeros(par.Nz)
        
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    def v_ss(self):
        """ social welfare in steady state """

        par = self.par
        ss = self.ss

        return np.sum([par.beta**t*ss.U_hh for t in range(par.T)])

    def v_path(self):
        """ social welfare in transition path """

        par = self.par
        path = self.path

        return np.sum([par.beta**t*path.U_hh[t] for t in range(par.T)])
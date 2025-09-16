
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks


def create_model(name,par=None,do_print=False):
    """ create model and solve for transition path to foreign shock """
    
    if do_print: print(name)
    if par is None: par = {}

    # a. create model
    if not 'HH_type' in par:
        model = HANKModelClass(name,par=par)
    else:
        if par['HH_type'] == 'HANK':
            model = HANKModelClass(name,par=par)
        elif par['HH_type'] == 'TANK':
            model = TANKModelClass(name,par=par)
        else:
            raise Exception("Wrong HH type chosen!")

    return model


class HANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def set_HH_type(self):

        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['w','r','L','LT', 'profits'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c', 'muc'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def set_blocks(self):
        self.blocks = [
            'blocks.production',
            'blocks.central_bank',
            'blocks.government',
            'hh',
            'blocks.labor_supply',
            'blocks.market_clearing'
        ]     


    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.set_HH_type()

        # c. GE
        self.shocks = ['epsT'] # exogenous inputs
        self.unknowns = ['pi','L', 'B', 'w'] # endogenous inputs
        self.targets = ['labor_supply_res','clearing_Y', 'gov_budget', 'NKPC_res'] # targets
        
        # d. all variables
        self.set_blocks()

        
    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. preferences
        par.beta = np.nan # discount factor (determined in ss)
        par.varphi = np.nan # disutility of labor (determined in ss)
        par.sigma = 1.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity

        # b. household type 
        par.HH_type = 'HANK' # HANK OR TANK 
        par.ann_mpc_target = 0.5 # MPC target 
        par.sHtM = par.ann_mpc_target # In TANK model set share of HtM HHs equal to MPC target

        # c. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.16 # std. of psi

        # d. price setting
        par.kappa = 0.03 # slope of Phillips curve
        par.mu = 1.1 # markup

        # e. government and central bank
        par.phi_Y = 0. # Taylor rule coefficient on output 
        par.tau = 0.33 # Tax rate
        par.H = 40 # Note: integer 
        
        par.B_target_ss = np.nan # bond supply
        par.r_target_ss = 0.02 # real interest rate

        # f. grids         
        self.set_HH_grids()

        # g. shocks
        par.jump_epsT = 0.01 # initial jump
        par.rho_epsT = 0.00 # AR(1) coefficeint
        par.std_epsT = 0.00 # std.

        # h. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

    def set_HH_grids(self):
        par = self.par 
        par.Nfix = 1
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points        
        par.Nz = 7 # number of productivity states

    def allocate(self):
        """ allocate model """
        par = self.par
        par.beta_grid = np.zeros(par.Nz) 
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss        


class TANKModelClass(HANKModelClass):

    def set_HH_type(self):

        self.pols_hh = []  # policy functions 
        self.grids_hh = []  # grids
        self.inputs_hh_z = []  # transition matrix inputs
        self.inputs_hh = []  # inputs to household problem
        self.intertemps_hh = []  # inputs to household problem
        self.outputs_hh = []  # output of household problem
        self.solve_hh_backwards = None    

    def set_HH_grids(self):

        par = self.par

        par.Nfix = 1
        par.a_min = np.nan # maximum point in grid for a
        par.a_max = np.nan# maximum point in grid for a
        par.Na = 0 # number of grid points        
        par.Nz = 0 # number of productivity states

    def set_blocks(self):
        self.blocks = [
            'blocks.production',
            'blocks.central_bank',
            'blocks.government',
            'blocks.TA_HHs',
            'blocks.labor_supply',
            'blocks.market_clearing'
        ]     
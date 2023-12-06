import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks

class IHANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['ra','inc_T','inc_NT'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','uc_T','uc_NT'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['ZT','ZNT','M_s','rF','PF_s'] # exogenous inputs
        self.unknowns = ['E','NNT','NT','piWT','piWNT'] # endogenous inputs
        self.targets = ['NKWCT_res','NKWCNT_res','clearing_YT','clearing_YNT','UIP_res'] # targets
        
        # d. all variables
        self.blocks = [
            'blocks.production',
            'blocks.prices',
            'blocks.inflation',
            'blocks.central_bank',
            'blocks.government',
            'hh',
            'blocks.NKWCs',
            'blocks.UIP',
            'blocks.consumption',
            'blocks.market_clearing',            
            'blocks.accounting',            
        ]        

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        
    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. discrete states
        par.Nfix = 2 # number of sectors sectors
        par.Nz = 7 # idiosyncratic productivity
        par.sT = 0.25 # share of workers in tradeable sector

        # b. preferences
        par.beta = 0.975 # discount factor
        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution

        par.alphaT = np.nan # share of tradeable goods in home consumption (determined in ss)
        par.etaT = 2.0 # elasticity of substitution between tradeable and non-tradeable goods
        
        par.alphaF = 1/3 # share of foreign goods in home tradeable consumption
        par.etaF = 2.0 # elasticity of substitution between home and foreign tradeable goods
          
        par.varphiT = np.nan # disutility of labor in tradeable sector (determined in s)
        par.varphiNT = np.nan # disutility of labor in non-tradeable sector (determined in s)
        par.nu = 1.0 # Frisch elasticity of labor supply
              
        # c. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of psi
        
        # d. price setting
        par.kappa = 0.1 # slope of wage Phillips curve
        par.muw = 1.2 # wage mark-up       
 
        # e. foreign Economy
        par.rF_ss = 0.005 # exogenous foreign interest rate
        par.eta_s = 2.0 # Armington elasticity of foreign demand
        par.M_s_ss = np.nan # size of foreign market (determined in ss)

        # f. government
        par.phi = 1.5 # Taylor rule coefficient on inflation
        par.tau_ss = 0.30 # tax rate on labor income
        par.omega = 0.10 # adjustment rate of taxes

        # g. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 50.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # h. shocks
        par.jump_M_s = 0.00 # initial jump
        par.rho_M_s = 0.00 # AR(1) coefficeint
        par.std_M_S = 0.00 # std.

        par.jump_rF = 0.00 # initial jump
        par.rho_rF = 0.00 # AR(1) coefficeint
        par.std_rF = 0.00 # std.

        par.jump_PF_s = 0.00 # initial jump
        par.rho_PF_s = 0.00 # AR(1) coefficeint
        par.std_PF_s = 0.00 # std.

        # i. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

    def allocate(self):
        """ allocate model """

        par = self.par
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss        
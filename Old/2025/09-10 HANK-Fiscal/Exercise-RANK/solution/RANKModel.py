import numpy as np
import numba as nb
from EconModel import EconModelClass
from GEModelTools import GEModelClass
import blocks 
import steady_state

class NKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ini','sim','ss','path']
        
        # b. other attributes (to save them)
        # self.other_attrs = ['grids_hh','pols_hh','inputs_hh','inputs_exo','inputs_endo','targets','varlist_hh','varlist','jac']

        # household
        self.grids_hh = [] # grids
        self.pols_hh = [] # policy functions
        self.inputs_hh_z = [] # transition matrix inputs

        self.inputs_hh = [] # inputs to household problem
        self.outputs_hh = [] # output of household problem
        self.intertemps_hh = []

        self.varlist_hh = [] # variables in household problem

        # GE
        self.shocks = ['Z', 'beta', 'eps_i', 'G'] # exogenous inputs 
        self.unknowns = ['C', 'P', 'Y', 'w'] # endogenous inputs
        self.targets = ['NKPC', 'Euler', 'goods_mkt', 'Labor_supply'] # targets
        self.blocks = ['blocks.NK_block']
        

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # Preferences
        par.CRRA = 2.0 # CRRA coefficient. Corresponds to EIS = 0.5
        par.inv_frisch = 2.0 # Frisch elasticity = 0.5 
        par.vphi = np.nan 
        par.betaF = np.nan 

        # Interest rate target 
        par.i_ss = 0.003 # target for real interest rate

        # Business cycle parameters 
        par.kappa = 0.05
        par.theta = 0.1 
        par.mu = 1.1 
        par.phi = 1.5      
        
        # ZLB
        par.ZLB = False 

        # Shock specifications 
        rho = 0.8
        for var in self.shocks:
            setattr(par,'jump_'+var, 0.01)
            setattr(par,'std_'+var, 0.001)
            setattr(par,'rho_'+var, rho)

        # Want monetary policy shock to be accomodating 
        par.jump_eps_i = -0.01
          
        # Misc.
        par.T = 300 # length of path - should be consistent with T in SOE model           
        par.simT = 100_000     
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-10 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-9 # tolerance when solving eq. system

        par.Nz = par.Nfix = 1        

        self.solve_hh_backwards = None
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # b. solution
        self.allocate_GE()


    find_ss = steady_state.find_ss            
            

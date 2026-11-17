import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks

class HANKModelClass(EconModelClass,GEModelClass):

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
        self.inputs_hh = ['Z','ra'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['eps_r'] # exogenous inputs
        self.unknowns = ['pi_w','L'] # endogenous inputs
        self.targets = ['NKWC_res','clearing_A'] # targets

        # d. all variables
        self.blocks = [
            'blocks.production',
            'blocks.central_bank',
            'blocks.government',
            'hh',
            'blocks.NKWC',
            'blocks.market_clearing'
        ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. preferences
        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity
        par.varphi = np.nan # disutility of labor (determined in ss)
        par.beta = np.nan # discount factor (determined in ss)

        # b. income
        par.rho_z = 0.95**4 # AR(1) parameter
        par.sigma_psi = 0.186 # std. of psi
        par.Nz = 7 # number of productivity states
        par.zeta = 0.0 # cyclicality of income shares: 0 acyclical, >0 procyclical risk
        par.Z_ss = np.nan # steady state income (determined in ss)

        # c. unions: no markups, so no stocks and households only hold bonds
        par.kappa = 0.16 # slope of wage Phillips curve

        # d. steady state targets
        par.r_target_ss = 0.02 # real interest rate
        par.B_target_ss = 0.25 # government debt = household wealth
        par.G = 0.20 # government spending, constant

        # e. grids
        par.Nfix = 1 # number of fixed types
        par.a_min = 0.0 # minimum point in grid for a
        par.a_max = 50.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # f. shock to the real rate rule
        par.jump_eps_r = -0.0025 # initial jump
        par.rho_eps_r = 0.4 # AR(1) coefficient

        # g. misc.
        par.T = 500 # length of path

    def allocate(self):
        """ allocate model """

        par = self.par
        par.z_ergodic = np.zeros(par.Nz)
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

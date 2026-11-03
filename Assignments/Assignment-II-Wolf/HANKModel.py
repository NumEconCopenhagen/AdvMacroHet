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
        self.inputs_hh = ['Z','ra','chi'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['G','eps_r','chi'] # exogenous inputs
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
        par.Nfix = 1
        par.beta = np.nan # discount factor (determined in ss)
        par.varphi = np.nan # disutility of labor (determined in ss)

        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity

        # b. income parameters
        par.rho_z = 0.95**4 # AR(1) parameter
        par.sigma_psi = 0.186 # std. of psi
        par.Nz = 7 # number of productivity states

        # c. price setting
        par.kappa = 0.16 # slope of wage Phillips curve
        par.mu_w = 1.2 # wage markup

        # d. government
        par.G_target_ss = 0.20 # government spending
        par.B_target_ss = 0.25 # bond supply
        par.r_target_ss = 0.02 # real interest rate

        # e. grids
        par.a_min = 0.0 # minimum point in grid for a
        par.a_max = 50.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # f. shocks
        par.jump_eps_r = -0.0025 # rate cut
        par.rho_eps_r = 0.40 # persistence

        # g. misc.
        par.T = 500 # length of path


    def allocate(self):
        """ allocate model """

        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

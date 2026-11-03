import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import blocks
import steady_state

class RANKModelClass(EconModelClass,GEModelClass):

    #########
    # setup #
    #########

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ini','sim','ss','path']

        # b. household (no heterogeneous household block)
        self.grids_hh = []
        self.pols_hh = []
        self.inputs_hh = []
        self.inputs_hh_z = []
        self.outputs_hh = []
        self.intertemps_hh = []

        # c. GE
        self.shocks = ['G']
        self.unknowns = ['Y','C_R']
        self.targets = ['clearing_Y','Euler']
        self.blocks = [
            'blocks.central_bank',
            'blocks.government',
            'blocks.households',
            'blocks.market_clearing',
            'blocks.unions',
            'blocks.fisher',
        ]

        self.solve_hh_backwards = None

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. households
        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.beta = np.nan # discount factor (determined in ss)
        par.lam = 0.0 # share of hand-to-mouth households (0 is RANK)
        par.nu = 1.0 # inverse Frisch elasticity
        par.varphi = np.nan # disutility of labor (determined in ss)

        # b. unions
        par.kappa = 0.16 # slope of wage Phillips curve

        # c. government
        par.r_target_ss = 0.02 # real interest rate
        par.G_target_ss = 0.20 # government spending
        par.B_target_ss = 0.25 # government debt
        par.rho_B = 0.9 # debt persistence (0 is balanced budget)

        # d. shocks
        par.jump_G = 0.01 # initial jump
        par.rho_G = 0.80**4 # AR(1) coefficient
        par.std_G = 0.0 # std.

        # e. misc.
        par.T = 200 # length of path
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

        par.Nz = par.Nfix = 1

    def allocate(self):
        """ allocate model """

        self.allocate_GE()

    find_ss = steady_state.find_ss

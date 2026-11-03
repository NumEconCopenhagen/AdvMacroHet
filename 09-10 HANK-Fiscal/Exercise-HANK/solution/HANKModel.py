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
        self.shocks = ['G'] # exogenous inputs
        self.unknowns = ['pi_w','L'] # endogenous inputs
        self.targets = ['NKWC_res','clearing_A'] # targets

        # d. all variables
        self.blocks = [
            'blocks.production',
            'blocks.central_bank',
            'blocks.government',
            'blocks.mutual_fund',
            'hh',
            'blocks.NKWC',
            'blocks.market_clearing'
        ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. model type: 'IKC' or 'canonical' (sets the calibration in allocate)
        par.model_type = 'canonical'

        # b. preferences
        par.Nfix = 1
        par.sigma = 1.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity
        par.varphi = np.nan # disutility of labor (determined in ss)

        par.Nbeta = 2 # number of discount factor states
        par.beta_L = 0.82 # low discount factor (set to get a high annual MPC)
        par.beta_H = np.nan # high discount factor (determined in ss)
        par.share_H = 0.49 # share of patient agents
        par.q = 0.04 # probability of a new draw of beta
        par.beta_avg = np.nan # average discount factor

        # c. income parameters
        par.rho_z = 0.95**4 # AR(1) parameter
        par.sigma_psi = 0.186 # std. of psi
        par.Nz_e = 7 # number of productivity states

        # d. firms and unions
        par.mu = np.nan # price markup (determined in ss)
        par.kappa = 0.16 # slope of wage Phillips curve

        # e. central bank
        par.taylor = False # False: real rate rule, True: Taylor rule
        par.phi_pi = 1.5 # Taylor rule coefficient on inflation

        # f. government
        par.nominal_bonds = False # False: real bonds, True: nominal bonds
        par.rho_B = 0.90 # persistence of excess debt

        # g. steady state targets
        par.r_target_ss = 0.02 # real interest rate
        par.A_target_ss = 5.0 # household wealth
        par.B_target_ss = 1.0 # government debt
        par.G_target_ss = 0.20 # government spending

        # h. grids
        par.a_min = 0.0 # minimum point in grid for a
        par.a_max = 250.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # i. shocks
        par.jump_G = 0.01 # initial jump
        par.rho_G = 0.80**4 # AR(1) coefficient
        par.std_G = 0.00 # std.

        # j. misc.
        par.T = 500 # length of path, long enough for persistent debt to be repaid
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. IKC: no stocks (A = B, so mu = 1), one discount factor, real rate rule and real bonds
        if par.model_type == 'IKC':
            par.sigma = 2.0
            par.A_target_ss = par.B_target_ss = 0.25
            par.a_max = 50.0
            par.Na = 500
            par.taylor = False
            par.nominal_bonds = False

        # b. allocate
        par.Nz = par.Nz_e*par.Nbeta
        par.beta_grid = np.zeros(par.Nz)
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

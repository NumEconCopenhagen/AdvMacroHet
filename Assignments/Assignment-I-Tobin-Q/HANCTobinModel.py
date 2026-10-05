import numpy as np

from consav.grids import nonlinspace
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCTobinModelClass(EconModelClass,GEModelClass):

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ini','sim','ss','path']

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['rB','cg','w','bequest'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = [] # exogenous shocks (only unexpected permanent changes are studied)
        self.unknowns = ['K'] # endogenous unknowns
        self.targets = ['clearing_A'] # targets = 0

        # d. blocks
        self.blocks = [
            'blocks.production_firm',
            'blocks.mutual_fund',
            'blocks.bequests',
            'hh',
            'blocks.government',
            'blocks.market_clearing']

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 5 # number of permanent income types
        par.Ne = 4 # number of idiosyncratic productivity states
        par.Nz = par.Ne*2 # idiosyncratic states times (alive, newborn)

        # a. preferences
        par.beta = 0.9 # discount factor [calibrated, initial guess]
        par.sigma = 2.0 # CRRA coefficient on consumption
        par.gamma = 2.0 # strength of taste for wealth [calibrated, initial guess]
        par.a_bar = 5.0 # shifter in taste for wealth [calibrated, initial guess]
        par.Sigma = np.nan # curvature of taste for wealth [calibrated]

        # b. demography
        par.xi = np.nan # death probability [calibrated]

        # c. labor productivity
        par.year = 1970 # year of the permanent income distribution
        par.rho_e = 0.95 # AR(1) parameter of idiosyncratic productivity
        par.sigma_e = 0.545 # std. of log idiosyncratic productivity (stationary)

        # d. production
        par.alpha = 1/3 # capital share
        par.mu = 1.2 # markup
        par.delta = np.nan # depreciation rate [calibrated]
        par.Gamma = np.nan # technology [calibrated]

        # e. government
        par.B = 0.19 # government debt
        par.tau_l = 0.30 # labor income tax
        par.tau_E = 0.30 # estate tax
        par.tau_d = 0.30 # dividend tax

        # f. portfolio rule: equity share of wealth (FITWI, SCF 1989-2019)
        par.portfolio_rule = True # if False, everybody holds the market portfolio
        par.psi0 = 1.0 # maximum equity share
        par.psi1 = 0.374 # slope of the equity share in log wealth
        par.psi2 = 2.598 # log relative wealth at which the equity share is half of the maximum
        par.A_ref = np.nan # reference wealth in the portfolio rule [calibrated]

        # g. calibration targets
        par.r_target = 0.03 # return on wealth
        par.WY_target = 3.7 # wealth-output ratio
        par.zeta_a_target = 1.35 # Pareto coefficient of wealth
        par.zeta_c_target = 3.02 # Pareto coefficient of consumption

        # h. grids
        par.a_max = 1e10 # maximum point in grid for a
        par.a_phi = 4.4 # curvature of the grid for a (finest around a = 5)
        par.Na = 500 # number of grid points

        # i. misc.
        par.T = 300 # length of transition path
        par.simT = 10 # length of simulation (not used)

        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system

        par.tol_solve = 1e-6 # tolerance when solving household problem (absolute, so loose: rounding at the top of the grid is about 1e-7)
        par.tol_simulate = 1e-17 # tolerance when simulating household problem (tight: the top of the distribution converges slowly)
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.Z_grid,par.share_grid = steady_state.permanent_income_grid(par.year)
        par.e_grid = np.zeros(par.Nz) # idiosyncratic productivity
        par.newborn_grid = np.zeros(par.Nz) # 1 if newborn
        par.exposure_grid = np.zeros(par.Na) # equity exposure s(a)a

        # b. solution
        self.allocate_GE()

        # c. asset grid
        par.a_grid[:] = nonlinspace(0.0,par.a_max,par.Na,par.a_phi)

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

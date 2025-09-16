import numpy as np

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state

class HANKSAMModelClass(EconModelClass,GEModelClass):    

    #########
    # setup #
    #########

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ini','ss','path','sim']
        
        # b. household
        self.grids_hh = ['a']
        self.pols_hh = ['a']
        self.inputs_hh = ['w','r','tau','div','transfer','u_bar'] # DELETE u_bar
        self.inputs_hh_z = ['delta','lambda_u_s']
        self.outputs_hh = ['a','c','u_ALL','u_UI','c_HtM','c_BS','c_PIH'] # DELTE c_HtM,c_BS,c_PIH   
        self.intertemps_hh = ['vbeg_a']

        # c. GE
        self.shocks = ['G','u_bar'] # DELETE u_bar
        
        self.unknowns = ['px',
                         'Vj',
                         'v',
                         'u',
                         'S',
                         'pi',
                         'U_UI_hh_guess']

        self.targets = ['errors_Vj',
                        'errors_Vv',
                        'errors_u',
                        'errors_pi',
                        'errors_assets',
                        'errors_U',
                        'errors_U_UI']

        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

        self.blocks = [
            'blocks.production',
            'blocks.labor_market',
            'blocks.entry',
            'blocks.price_setters',
            'blocks.central_bank',
            'blocks.dividends',
            'blocks.financial_market',
            'blocks.government',
            'hh',
            'blocks.market_clearing',
            'blocks.ann']

    def setup(self):
        """ set baseline parameters """

        par = self.par
        par.Nfix = 3 # number of household types

        # a. consumption-saving
        par.r_ss = 1.02**(1/12) - 1 # real interest rate in steady state

        par.beta_HtM = 0.0 # discount factor of HtM households
        par.beta_BS = 0.940**(1/12) # discount factor of buffer-stock households
        par.beta_PIH = 0.975**(1/12) # discount factor of permanent income hypothethis households
        par.beta_firm = 0.975**(1/12) # discount factor of firms

        par.HtM_share = 0.30 # share of HtM households
        par.PIH_share = 0.10 # share of PIH households
        # the share of buffer-stock households is the residual 1-par.HtM_share-par.PIH_share
        
        par.sigma = 2.0 # CRRA coefficient

        # b. matching and bargaining
        par.A = np.nan # matching efficiency, determined endogenously
        par.theta_ss = 0.60 # tightness in ss
        par.alpha = 0.60 # matching elasticity
        par.lambda_u_s_ss = 0.30 # job-finding rate in steady state (per effective seacher)
        
        # c. intermediary goods firms
        par.w_share_ss = 0.90 # wage in steady state
        par.delta_ss = 0.020 # separation rate in steady state
        par.kappa = np.nan # flow vacancy cost, determined endogenously

        # d. whole-sale and final-good firms
        par.epsilon = 6.0 # price elasticity      
        par.phi = 600.0 # Rotemberg cost

        # e. monetary policy
        par.delta_pi = 1.5 # inflation aggressiveness

        # f. government
        par.tau_ss = 0.30 # tax rate in steady state
        par.phi_obar = 0.70 # high unemployment insurance
        par.phi_ubar = 0.40 # low unemployment insurance
        par.u_bar_ss = 6.0 # max duration for high unemployment insurance
        
        par.delta_q = 1-1/36 # maturity of government bonds
        par.omega = 0.05 # responsiveness of tax to debt

        # g. shocks
        par.rho_G = 0.80 # persistence of government spending
        par.jump_G = np.nan # jump (determined in ss)

        # h. household problem
        par.Nu = 10 # number of unemployment states
        par.Na = 300 # number of asset grid points
        par.a_max = 200 # max level of assets

        # i. misc
        par.T = 12*40 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 50 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

        par.py_hh = False
        par.py_blocks = False
        par.full_z_trans = True

        ## DELETE ##
        # j. extensions
        par.RA = False
        ## DELETE ##

    def allocate(self):
        """ allocate model """
        
        self.create_grids()
        self.allocate_GE()

    def create_grids(self):
        """ create grids """

        par = self.par

        # a. z
        par.Nz = par.Nu+1

        par.i_u_hh = np.zeros((par.Nfix,par.Nz))
        par.i_u_hh[:,1:] = np.arange(1,par.Nu+1)

        # b. beta
        par.beta_grid = np.nan*np.ones(par.Nfix)
        par.beta_shares = np.nan*np.ones(par.Nfix)

    def calc_Cs(self,i_fix=None):
        """ calculate consumption """

        par = self.par
        ss = self.ss

        if i_fix is None:
            i = 0
            j = par.Nfix
        else:
            i = i_fix
            j = i + 1
        
        C_u_dur = np.nan*np.ones(par.Nu)

        if np.sum(ss.D[i:j]) > 0.0:

            C_e = np.sum(ss.D[i:j,0,:]*ss.c[i:j,0,:])/np.sum(ss.D[i:j,0,:])
            C_u = np.sum(ss.D[i:j,1:,:]*ss.c[i:j,1:,:])/np.sum(ss.D[i:j,1:,:])

            for i_z in range(par.Nz):

                i_u = par.i_u_hh[i:j,i_z]
                C_u_dur[int(i_u[0]-1)] = np.sum(ss.D[i:j,i_z,:]*ss.c[i:j,i_z,:])/np.sum(ss.D[i:j,i_z,:])
        
        else:

            C_e = np.nan
            C_u = np.nan

        return C_e,C_u,C_u_dur

    # DELTE BELOW HERE

    def run(self,find_ss=True,compute_jacs=True,skip_hh=False,show_outcomes=True):
        """ run model """

        if find_ss: self.find_ss()
        if compute_jacs: self.compute_jacs(skip_hh=skip_hh,skip_shocks=True)
        self.find_transition_path(shocks=['G'],do_end_check=False)

        self.calc_outcomes(do_print=show_outcomes)

    def calc_outcomes(self,do_print=False):
        """ calculate outcomes """

        self.outcomes = {}
        
        self.outcomes['IMPCs'] = self.IMPCs()
        self.outcomes['M'] = self.fiscal_multiplier()

        if do_print:
            print(f' qB  = {self.ss.qB:.2f}')
            print(f' M   = {self.outcomes["M"]:.2f}')
            for k in [0,1,2,12]:
                print(f' MPC[{k:2d}] = {self.outcomes["IMPCs"][k]*100:.1f}')

    def fiscal_multiplier(self):
        """ calculate fiscal multiplier """

        par = self.par
        ss = self.ss
        path = self.path

        nom = np.sum([(path.Y[t]-ss.Y)/(1+ss.r)**t for t in range(par.T)])
        denom = np.sum([(path.taxes[t]-ss.taxes)/(1+ss.r)**t for t in range(par.T)])

        return nom/denom
    
    def IMPCs(self):
        """ calculate IMPCs """

        par = self.par
        ss = self.ss

        dy = 0.01*(1-ss.tau)*ss.w
        custom_paths = {'transfer':ss.transfer*np.ones(par.T)}
        custom_paths['transfer'][0] += dy

        path = self.decompose_hh_path(do_print=False,use_inputs=['transfer'],custom_paths=custom_paths);
        
        return (path.C_hh[:,0]-ss.C_hh)/dy
    
    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

## DELETE ##
class HANKSAMModelClass_alt(HANKSAMModelClass):

    def settings(self):

        super().settings()
        
        self.unknowns = ['px',
                         'Vj',
                         'v',
                         'S',
                         'pi',
                         'U_UI_hh_guess']

        self.targets = ['errors_Vj',
                        'errors_Vv',
                        'errors_pi',
                        'errors_assets',
                        'errors_U',
                        'errors_U_UI']

        self.blocks = [
            'blocks.production',
            'blocks.labor_market_alt',
            'blocks.entry',
            'blocks.price_setters',
            'blocks.central_bank',
            'blocks.dividends',
            'blocks.financial_market',
            'blocks.government',
            'hh',
            'blocks.market_clearing',
            'blocks.ann']
        
## DELETE ##
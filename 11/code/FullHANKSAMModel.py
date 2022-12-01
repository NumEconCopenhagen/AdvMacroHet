import pickle
from copy import deepcopy
import time
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from EconModel import EconModelClass
from GEModelTools import GEModelClass
from consav.misc import elapsed

import household_problem
import steady_state
import blocks

from root_finding import brentq

class FullHANKSAMModelClass(EconModelClass,GEModelClass):    

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
        self.inputs_hh = ['shock_beta','w','RealR_ex_post','tau','u_bar','phi_obar','transfer']
        self.inputs_hh_z = ['delta','lambda_u']
        self.outputs_hh = ['a','c','y','u_UI']
        self.intertemps_hh = ['vbeg_a']

        # c. GE
        self.shocks = ['shock_TFP','shock_beta','u_bar','phi_obar',
            'wage_subsidy','hiring_subsidy','public_transfer','G']
        self.unknowns = ['px','Vj','Vv','Pi','ut','vt','U_UI_hh_guess']
        self.targets = ['errors_Vj','errors_Vv','errors_Pi','errors_assets','errors_ut','errors_vt','errors_U_UI']

        # d. all variables
        self.varlist = []
        self.varlist += ['v','u','delta','entry','vt','ut']
        self.varlist += ['theta','lambda_u','lambda_v','w']
        self.varlist += ['M','mu','Vj','Vv','shock_TFP']
        self.varlist += ['px','R','Pi','RealR','RealR_ex_post','shock_beta','A_hh','C_hh']
        self.varlist += ['tau','phi_obar','u_bar','U_UI_hh_guess']
        self.varlist += ['B','q','qB','Yt_hh','UI']
        self.varlist += ['errors_Vj','errors_Vv','errors_Pi','errors_assets',
                         'errors_WageRule','errors_ut','errors_vt','errors_U_UI']
        self.varlist += ['wage_subsidy','hiring_subsidy','transfer','public_transfer','G']

        # functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.fill_z_trans = household_problem.fill_z_trans
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post

        # misc
        self.other_attrs = ['data','moms','datamoms']

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. macros
        par.wage_setting = 'fixed'
        par.free_entry = False
        par.exo_sep = False

        par.only_SAM = False
        par.var_u_target = np.nan

        # b. model parameters

        # preferences
        par.beta_grid = np.array([0.00,0.96**(1/12),0.98**(1/12)]) # discount factors
        
        par.HtM_share = 0.15 # low beta share
        par.PIH_share = 0.15 # high beta share
        
        par.beta_firm = par.beta_grid[-1] # firm discount factor
        par.beta_HtM = 0.90 # cut-off for hands-to-mouth bheavior#

        par.sigma = 2.0 # CRRA coefficient

        # matching and bargaining
        par.A = np.nan # matching efficiency, determined endogenously
        par.theta_ss = 0.60 # tightness in ss
        par.alpha = 0.60 # matching elasticity
        par.rho_w = 0.00 # wage rule, elasticity
        par.eta_u = 0.00 # wage rule, elasticity
        par.eta_e = 0.00 # wage rule, elasticity
        par.eta_TFP = 0.00 # wage rule, elasticity

        # intermediary goods firms
        par.w_ss = 0.70 # wage in steady state
        par.kappa = np.nan # flow vacancy cost, determined endogenously
        par.kappa_0 = 0.1 # fixed vacancy cost
        par.psi = 1.0 # separation elasticity
        par.xi = 0.02 # entry elasticity
        par.p_fac = 1.20 # factor for maximum increase in separation rate
        par.p = np.nan # maximum separation rate, determined endogenously
        par.Upsilon = np.nan # Vj at maximum separation rate, determined endogenously

        # final goods firms
        par.epsilon_p = 6.0 # price elasticity      
        par.phi = 600.0 # Rotemberg cost

        # monetary policy
        par.rho_R = 0.0 # inertia
        par.delta_pi = 1.5 # inflation aggressiveness

        # government
        par.qB_share_ss = 1.00 # government bonds (share of wage)

        par.Nu = 13 # number of u states
        par.u_bar_ss = 6.0 # UI duration
        par.phi_obar_ss = 0.76 # high UI ratio (rel. to w) *before* exhausation (in steady state)
        par.phi_ubar = 0.55 # low UI ratio (rel. to w) *after* exhausation
        par.UI_prob = 0.50 # probability of getting UI

        par.omega = 0.90 # responsiveness of tax to debt
        par.delta_q = 1-1/60 # maturity of government bonds

        # idiosyncratic productivity
        par.Ne = 1
        par.rho_e = 0.95
        par.sigma_psi = 0.10

        # b. shocks
        par.rho_shock_TFP = 0.965 # persitence
        par.jump_shock_TFP = -0.007 # jump

        par.rho_shock_beta = 0.965 # persistence
        par.jump_shock_beta = 0.0 # jump

        # c. household problem
        par.div_hh = 0.0 # share of dividends distributed to households
        par.Na = 500 # number of asset grid points
        par.a_max = 50 # max level of assets

        # d. misc
        par.T = 300 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 50 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-8 # tolerance when solving eq. system
        par.tol_R = 1e-12 # tolerance when finding RealR for ss
        par.tol_calib = 1e-5 # tolerance when calibrating (C_drop and var_u)
        
    def allocate(self):
        """ allocate model """
        
        par = self.par

        # implied grids and grid sizes
        par.beta_shares = np.array([par.HtM_share,0.0,par.PIH_share])
        par.beta_shares[1] = 1-np.sum(par.beta_shares)
        assert par.beta_shares[1] >= 0.0

        par.Nfix = par.beta_grid.size
        assert par.beta_shares.size == par.Nfix

        par.e_grid = np.zeros(par.Ne)
        par.Nz = (par.Nu+1)**par.Ne
        par.i_u_hh = np.tile(np.repeat(np.arange(par.Nu+1),par.Ne),par.Nfix).reshape((par.Nfix,-1))
        
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    ##############
    # more setup #
    ##############

    def load_data(self):
        """ load data """

        with open('data/data.p', 'rb') as f:
            self.data = pickle.load(f)

    def load_parameters(self,name,varlist=None):
        """ load parameters from file """

        # a. load
        with open(f'saved/par_{name}.p', 'rb') as f:
            par_dict_loaded = pickle.load(f)

        # b. update parameters
        if varlist is None: varlist = list(par_dict_loaded.keys())
        for key in varlist:
            self.par.__dict__[key] = par_dict_loaded[key] 

    def set_macros(self,free_entry=None,wage_setting=None):
        """ set macros """

        par = self.par

        # baseline
        unknowns = ['px','Vj','Vv','Pi','ut','vt','U_UI_hh_guess']
        targets = ['errors_Vj','errors_Vv','errors_Pi','errors_assets','errors_ut','errors_vt','errors_U_UI']

        if not free_entry is None: par.free_entry = free_entry
        if not wage_setting is None: par.wage_setting = wage_setting 

        # a. free entry
        if par.free_entry:
            unknowns = ['px','entry','Vj','Pi','ut','vt','U_UI_hh_guess']

        # b. wage setting
        if par.wage_setting == 'fixed':
            pass
        elif par.wage_setting == 'rule':
            unknowns += ['w']
            targets += ['errors_WageRule']
        else:
            raise NotImplementedError

        self.update_aggregate_settings(unknowns=unknowns,targets=targets)

    def set_only_SAM(self):
        """ set only SAM block active """

        par = self.par

        # a. no households
        par.only_SAM = True 
        self.grids_hh = []
        self.pols_hh = []
        self.inputs_hh = []
        self.outputs_hh = []        

        # b. simpler unknowns and targets
        shocks = ['shock_TFP','px','u_bar','phi_obar']
        targets = ['errors_Vj','errors_Vv','errors_ut','errors_vt']

        if par.free_entry:
            unknowns = ['Vj','entry','ut','vt']
        else:
            unknowns = ['Vj','Vv','ut','vt']

        if par.wage_setting == 'fixed':
            pass
        elif par.wage_setting == 'rule':
            unknowns += ['w']
            targets += ['errors_WageRule']
        else:
            raise NotImplementedError

        self.update_aggregate_settings(shocks=shocks,unknowns=unknowns,targets=targets)

    ###############
    # calibration #
    ###############

    def get_IRF(self,varname):
        """ get IRF from model or data """

        # a. steady state
        ssvalue = getattr(self.ss,varname)

        # b. values
        values = getattr(self.path,varname)[0,:]

        # c. transformation
        if varname in ['R','RealR','Pi']:

            IRF = 100*(values**12-ssvalue**12)
            ylabel = '%-points (ann.)'

        elif varname in ['u','delta','lambda_u','lambda_v','shock_beta']:

            IRF = 100*(values-ssvalue)
            ylabel = '%-points'

        elif varname in ['u_bar']:

            IRF = values-ssvalue
            ylabel = 'months'

        else:

            IRF = 100*(values - ssvalue)/np.abs(ssvalue)
            ylabel = '%'

        # d. return
        return IRF,ylabel

    def calc_calib_moms_ss(self,do_print=False):
        
        par = self.par
        ss = self.ss
        moms = self.moms
        datamoms = self.datamoms = {}

        if not par.only_SAM:

            C_e = np.sum(ss.D[:,0,:]*ss.c[:,0,:])/np.sum(ss.D[:,0,:])
            C_u = np.sum(ss.D[:,1:,:]*ss.c[:,1:,:])/np.sum(ss.D[:,1:,:])
            C_u_ini = np.sum(ss.D[:,1,:]*ss.c[:,1,:])/np.sum(ss.D[:,1,:])

            moms['C_drop_ss'] = (C_u/C_e-1)*100
            moms['C_drop_ini_ss'] = (C_u_ini/C_e-1)*100

        # d. print
        if do_print:
            for k,v in moms.items():
                print(f'{k:10s} = {v:7.4f}')        
            print('\ndata:')
            for k,v in datamoms.items():
                print(f'{k:10s} = {v:7.4f}')        

    def calc_calib_moms(self,do_print=False):
        """ calculate moments for calibration """

        par = self.par
        ss = self.ss
        path = self.path

        moms = self.moms = {}
        datamoms = self.datamoms = {}

        # a. IRFs
        u_IRF,_ = self.get_IRF('u')
        delta_IRF,_ = self.get_IRF('delta')
        lambda_u_IRF,_ = self.get_IRF('lambda_u')

        # b. moments
        delta = ss.delta + delta_IRF/100
        lambda_u = ss.lambda_u + lambda_u_IRF/100

        u_approx = 100*(1-lambda_u)*delta/((1-lambda_u)*delta+lambda_u) - 100*ss.u
        u_approx_EU = 100*(1-ss.lambda_u)*delta/((1-ss.lambda_u)*delta+ss.lambda_u) - 100*ss.u
        u_approx_UE = 100*(1-lambda_u)*ss.delta/((1-lambda_u)*ss.delta+lambda_u) - 100*ss.u

        moms['w_share'] = 100*ss.w/(ss.px*ss.shock_TFP)
        moms['M_share'] = 100*ss.M/(ss.px*ss.shock_TFP)
        moms['var_u'] = np.sum((100*(path.u[0,:]-ss.u))**2)
        if np.any(path.w[0,:] <= 0):
            moms['std_W'] = np.nan
        else:
            moms['std_W'] = np.sqrt(np.sum(((np.log(path.w[0,:])-np.log(ss.w)))**2))
        moms['timeshift'] = np.argmax(np.abs(lambda_u_IRF))-np.argmax(np.abs(delta_IRF))       
        moms['EU_share'] = 100*np.cov(u_approx,u_approx_EU)[0,1]/np.var(u_approx)

        if not par.only_SAM:
            self.calc_calib_moms_ss(do_print=False)

        # c. data moments
        I = self.data['EU'].notna() & self.data['UE'].notna() & self.data['u'].notna()
        cycle_u,_ = sm.tsa.filters.cffilter(np.log(self.data['u'][I]),12,np.inf,drift=False)
        ss_u = np.mean(self.data['u'][I])
        u = ss_u + (cycle_u-np.mean(cycle_u))*ss_u

        datamoms['var_u'] = np.var(u)

        # d. print
        if do_print:
            for k,v in moms.items():
                print(f'{k:10s} = {v:7.4f}')        
            print('\ndata:')
            for k,v in datamoms.items():
                print(f'{k:10s} = {v:7.4f}')        

    def obj_calib_var_u(self,w_ss,do_print=False):
        """ objective when calibrating w_ss """
        
        par = self.par
        par.w_ss = w_ss

        if do_print: print('')

        t0 = time.time()
        self.find_ss()
        if do_print: print(f'              find_ss: {elapsed(t0)}')

        t0 = time.time()
        skip_hh = len(self.outputs_hh) == 0
        inputs_hh_all = [x for x in self.inputs_hh if not x in ['shock_beta','w']]
        self.compute_jacs(skip_hh=skip_hh,inputs_hh_all=inputs_hh_all,skip_shocks=True)
        if do_print: print(f'         compute_jacs: {elapsed(t0)}')

        t0 = time.time()
        self.find_transition_path()
        if do_print: print(f' find_transition_path: {elapsed(t0)}')

        t0 = time.time()
        self.calc_calib_moms()
        if do_print: print(f'      calc_calib_moms: {elapsed(t0)}')

        if do_print: print('')
        
        if np.isnan(par.var_u_target):
            return self.moms['var_u']-self.datamoms['var_u']
        else:
            return self.moms['var_u']-par.var_u_target

    def find_search_bracket(self,x_min,x_max,max_iter=100,do_print=False,do_full_print=False):
        """ find bracket to search for w_ss in """

        # a. at user-defined maximum
        var_u_diff = self.obj_calib_var_u(x_min,do_print=do_full_print)
        if do_print: print(f'w_ss = {x_min:12.8} -> var_u-var_u_data = {var_u_diff:12.8f}')

        assert np.sign(var_u_diff) < 0, 'must have var_u < var_u_data at the minimum value for w_ss'

        # b. find bracket      
        lower = x_min
        upper = x_max

        fupper = np.nan
        flower = var_u_diff

        it = 0
        while it < max_iter:
                    
            # i. midpoint and value
            x = (lower+upper)/2 # midpoint
            try:
                var_u_diff = self.obj_calib_var_u(x,do_print=do_full_print)
            except Exception as e:
                if do_full_print: print('')
                var_u_diff = np.nan

            if do_print: print(f'w_ss = {x:12.8} -> var_u-var_u_data = {var_u_diff:12.8f}')

            # ii. check conditions
            valid = not np.isnan(var_u_diff)
            correct_sign = np.sign(var_u_diff) > 0
            
            # iii. next step
            if valid and correct_sign: # found!
                upper = x
                fupper = var_u_diff
                return lower,upper,flower,fupper
            elif not valid: # too low s -> increase lower bound
                upper = x
            else: # too high s -> increase upper bound
                lower = x
                flower = var_u_diff

            # iv. increment
            it += 1

        raise ValueError('cannot find bracket for w_ss')

    def calibrate_to_var_u(self,do_print=False,do_full_print=False):
        """ calibrate to fit unemployment variance """
        
        t0 = time.time()

        par = self.par

        # a. find bracket to seach
        if do_print: print(f'find bracket to search in:')
        w_ss_min = 0.5
        w_ss_max = self.ss.px
        a,b,fa,fb = self.find_search_bracket(w_ss_min,w_ss_max,
            do_print=do_print,do_full_print=do_full_print)

        if do_print:  print(f'w_ss in [{a:.4f},{b:.4f}]')

        # b. search
        if do_print: print(f'brentq:')
        f = lambda x: self.obj_calib_var_u(x,do_print=do_full_print)
        root,_ = brentq(f,a,b,xtol=par.tol_calib,rtol=par.tol_calib,fa=fa,fb=fb,
            do_print=do_print,varname='w_ss',funcname='var_u-var_u_data')

        # c. final evaluation
        if do_print: print(f'final evaluation:')
        f(root)
        
        if do_print: print(f'calbiration done in {elapsed(t0)}')

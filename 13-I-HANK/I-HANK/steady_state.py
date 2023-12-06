# find steady state
import time
import numpy as np
from scipy import optimize

import blocks
from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. z
    par.z_grid[:],ss.z_trans[:,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        
        if i_fix == 0:
            ss.Dbeg[i_fix,:,0] = e_ergodic*par.sT
        elif i_fix == 1:
            ss.Dbeg[i_fix,:,0] = e_ergodic*(1-par.sT)
        else:
            raise NotImplementedError('i_fix must be 0 or 1')
        
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]

            if i_fix == 0:
                inc = ss.inc_T*z
            elif i_fix == 1:
                inc = ss.inc_NT*z

            c = (1+ss.ra)*par.a_grid + inc
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. prices

    # normalzied to 1
    for varname in ['PF_s','E','PF','PNT','PTH','PT','P','PTH_s','Q']:
        ss.__dict__[varname] = 1.0
    
    # zero inflation
    for varname in ['pi_F_s','pi_F','pi_NT','pi_TH','pi_T','pi','pi_TH_s','piWT','piWNT']:
        ss.__dict__[varname] = 0.0

    # real+nominal interest rates are equal to foreign interest rate
    ss.ra = ss.r  = ss.i = ss.rF = par.rF_ss
    ss.UIP_res = 0.0

    # b. production

    # normalize TFP and labor
    ss.ZT = 1.0
    ss.ZNT = 1.0
    ss.NT = 1.0*par.sT
    ss.NNT = 1.0*(1-par.sT)
    
    # production
    ss.YT = ss.ZT*ss.NT
    ss.YNT = ss.ZNT*ss.NNT

    # real = nominal wages = value of mpl
    ss.wT = ss.WT = ss.PTH*ss.ZT
    ss.wNT = ss.WNT = ss.PNT*ss.ZNT
    
    # c. household 
    ss.tau = par.tau_ss
    ss.inc_T = (1-ss.tau)*ss.wT*ss.NT/par.sT
    ss.inc_NT = (1-ss.tau)*ss.wNT*ss.NNT/(1-par.sT)

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # d. government
    ss.B = ss.A_hh
    ss.G = ss.tau*(ss.wT*ss.NT+ss.wNT*ss.NNT)-ss.r*ss.B
    
    # e. consumption

    # tradeables vs. non-tradeables
    par.alphaT = 1-(ss.YNT-ss.G)/ss.C_hh # clearing_NT

    ss.CT = par.alphaT*ss.C_hh 
    ss.CNT = (1-par.alphaT)*ss.C_hh

    # home vs. foreign
    ss.CTH = (1-par.alphaF)*ss.CT
    ss.CTF = par.alphaF*ss.CT

    # size of foreign market
    ss.CTH_s = ss.M_s = ss.YT - ss.CTH # clearing_T

    # f. market clearing
    ss.clearing_YT = ss.YT - ss.CTH - ss.CTH_s 
    ss.clearing_YNT = ss.YNT - ss.CNT - ss.G

    # zero net foreign assets
    ss.NFA = ss.A_hh - ss.B

    # zero net foreign assets
    ss.GDP = ss.YT + ss.YNT
    ss.NX = ss.GDP - ss.C_hh - ss.G
    ss.NFA = ss.A_hh - ss.B
    ss.CA = ss.NX + ss.ra*ss.NFA
    ss.Walras = ss.CA

    # g. disutility of labor for NKWPCs
    par.varphiT = 1/par.muw*(1-ss.tau)*ss.wT*ss.UC_T_hh/par.sT / ((ss.NT/par.sT)**par.nu)
    par.varphiNT = 1/par.muw*(1-ss.tau)*ss.wNT*ss.UC_NT_hh/(1-par.sT) / ((ss.NNT/(1-par.sT))**par.nu)
    ss.NKWCT_res = 0.0
    ss.NKWCNT_res = 0.0

def find_ss(model, do_print=False): 
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    evaluate_ss(model,do_print=do_print)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.inc_T = :.3f}')
        print(f'{ss.inc_NT = :.3f}')
        print(f'{par.alphaT = :.3f}')
        print(f'{par.alphaF = :.3f}')
        print(f'{par.varphiT = :.3f}')
        print(f'{par.varphiNT = :.3f}')
        print(f'{ss.M_s = :.3f}')
        print(f'{ss.clearing_YT = :12.8f}')
        print(f'{ss.clearing_YNT = :12.8f}')
        print(f'{ss.G = :.3f}')
        print(f'{ss.NFA = :.3f}')

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec
from tauchen import tauchen_trans_nb

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,z_scale,z,upsilon,u,ss=False):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # par,z_trans: always inputs
    # r,w: inputs because they are in .inputs_hh
    # vbeg_a, vbeg_a_plus: input because vbeg_a is in .intertemps_hh
    # a,c,l: outputs because they are in .outputs_hh

    # ss = True is to get guess of vbeg_a
    
    # a. update z_trans 
    sigma_psi_ = par.sigma_psi*upsilon
    z_trans_ = tauchen_trans_nb(par.z_log_grid,0.0,par.rho_z,sigma_psi_)
    for i_fix in nb.prange(par.Nfix):
        z_trans[i_fix] = z_trans_

    # b. EGM
    for i_fix in nb.prange(par.Nfix): # fixed types
        
        # i. solve
        for i_z in nb.prange(par.Nz): # stochastic discrete states
            
            # o. cash-on-hand
            m = (1+r)*par.a_grid + w*par.z_grid[i_z]*z_scale

            # oo. invert Euler and budget constraint
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid # current consumption + end-of-period assets
            
            # ooo. interpolation to fixed grid
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            
            c[i_fix,i_z] = m-a[i_fix,i_z]
            z[i_fix,i_z] = z_scale*par.z_grid[i_z]

        # ii. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a
    
    # c. utility
    u[:] = c**(1-par.sigma)/(1-par.sigma)
# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit       
def solve_hh_backwards(par,z_trans,r,L,tax,transfers,vbeg_a_plus,vbeg_a,a,c,muc):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """
    income = 0.0

    # determine number of low-productivity states that receive targeted transfers
    n_low_states = par.Nz
    low_mass = 1.0
    if par.low_transfers:
        cum_share = 0.0
        n_low_states = 0
        for i_z in range(par.Nz):
            cum_share += par.e_ergodic[i_z]
            n_low_states += 1
            if cum_share >= par.share_low:
                break
        low_mass = cum_share

    for i_fix in range(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
        
            z = par.z_grid[i_z]
            income = (1-tax)*(L*z)
            if par.low_transfers:
                if i_z < n_low_states and low_mass > 0.0:
                    income += transfers/low_mass
            else:
                income += transfers
    
            # i. EGM
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid
            
            # ii. interpolation to fixed grid
            m = (1+r)*par.a_grid + income
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z]
        
        # b. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a
        muc[i_fix] = c[i_fix]**(-par.sigma)

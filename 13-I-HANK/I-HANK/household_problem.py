# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit       
def solve_hh_backwards(par,z_trans,ra,inc_TH,inc_NT,vbeg_a_plus,vbeg_a,a,c,uc_TH,uc_NT):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            # a. solve step

            # i. income
            if i_fix == 0:
                inc = inc_TH/par.sT
            else:
                inc = inc_NT/(1-par.sT)
         
            z = par.z_grid[i_z]

            # ii. EGM
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid
            
            # iii. interpolation to fixed grid
            m = (1+ra)*par.a_grid + inc*z
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z]

        # b. expectation step
        v_a = (1+ra)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

    # extra output
    uc_TH[:] = 0.0
    uc_NT[:] = 0.0

    for i_z in range(par.Nz):
        uc_TH[0,i_z,:] = c[0,i_z,:]**(-par.sigma)*par.z_grid[i_z]
        uc_NT[1,i_z,:] = c[1,i_z,:]**(-par.sigma)*par.z_grid[i_z]

    uc_TH[:] /= par.sT
    uc_NT[:] /= (1-par.sT)
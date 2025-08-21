import numpy as np
import numba as nb
from consav.linear_interp import interp_1d_vec


@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,
                       returns,ss=False):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # par,z_trans: always inputs
    # r,w: inputs because they are in .inputs_hh
    # vbeg_a, vbeg_a_plus: input because vbeg_a is in .intertemps_hh
    # a,c,l: outputs because they are in .outputs_hh

    # ss = True is to get guess of vbeg_a

   

    for i_fix in nb.prange(par.Nfix): # fixed types

        # a. solve step
        for i_z in nb.prange(par.Nz): # stochastic discrete states

            ## i. cash-on-hand
            m = (1+r)*par.a_grid + w*par.e_grid[i_z]
            returns[i_fix,i_z,:] = r*par.a_grid

            if ss:

                a[i_fix,i_z,:] = 0.01

            else:

                # ii. EGM
                c_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
                m_endo = c_endo + par.a_grid # current consumption + end-of-period assets
                
                # iii. interpolation to fixed grid
                interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
                a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            
            c[i_fix,i_z] = m - a[i_fix,i_z]

        # b. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a
        

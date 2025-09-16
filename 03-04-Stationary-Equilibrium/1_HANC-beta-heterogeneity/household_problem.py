import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # a. solve step
    for i_fix in range(par.Nfix):
        for i_z in nb.prange(par.Nz):

            # i. back out consumption from Va
            c_endo = (par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)

            # ii. compute endogenous grid of assets
            a_endo = (c_endo + par.a_grid - w * par.z_grid[i_z]) / (1+r)
            
            # ii. interpolation to fixed grid
            interp_1d_vec(a_endo,par.a_grid,par.a_grid,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = par.a_grid * (1+r) + w * par.z_grid[i_z] -a[i_fix,i_z]

        # b. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

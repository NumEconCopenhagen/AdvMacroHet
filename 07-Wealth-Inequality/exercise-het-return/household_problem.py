import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,returns):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # TODO: add the excess return state, the code below is the standard model with productivity only
    for i_z in range(par.Nz):

        # a. EGM
        c_endo = (par.beta*vbeg_a_plus[0,i_z])**(-1/par.sigma)
        a_endo = (c_endo + par.a_grid - w*par.e_grid[i_z])/(1+r)
        interp_1d_vec(a_endo,par.a_grid,par.a_grid,a[0,i_z])
        a[0,i_z,:] = np.fmax(a[0,i_z,:],0.0)
        c[0,i_z] = (1+r)*par.a_grid + w*par.e_grid[i_z] - a[0,i_z]

        # b. capital income
        returns[0,i_z] = r*par.a_grid

    # c. envelope condition and expectation step
    v_a = (1+r)*c[0]**(-par.sigma)
    vbeg_a[0] = z_trans[0]@v_a

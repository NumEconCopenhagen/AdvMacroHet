import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,returns):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    v_a = np.zeros((par.Nz,par.Na))

    for i_z in range(par.Nz):

        # a. excess return and productivity states (r first in the Kronecker product)
        i_r = i_z // par.Ne
        i_e = i_z % par.Ne
        r_eff = r + par.r_grid[i_r]

        # b. EGM
        c_endo = (par.beta*vbeg_a_plus[0,i_z])**(-1/par.sigma)
        a_endo = (c_endo + par.a_grid - w*par.e_grid[i_e])/(1+r_eff)
        interp_1d_vec(a_endo,par.a_grid,par.a_grid,a[0,i_z])
        a[0,i_z,:] = np.fmax(a[0,i_z,:],0.0)
        c[0,i_z] = (1+r_eff)*par.a_grid + w*par.e_grid[i_e] - a[0,i_z]

        # c. capital income and envelope condition
        returns[0,i_z] = r_eff*par.a_grid
        v_a[i_z] = (1+r_eff)*c[0,i_z]**(-par.sigma)

    # d. expectation step
    vbeg_a[0] = z_trans[0]@v_a

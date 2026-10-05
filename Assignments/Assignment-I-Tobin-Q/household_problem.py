import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec


@nb.njit(parallel=True)
def solve_hh_backwards(par, z_trans, rB, cg, w, bequest, vbeg_a_plus, vbeg_a, a, c, ss=False):
    """ one EGM step with taste for wealth and death """

    # a. marginal utility of end-of-period wealth on the grid
    v_wealth = par.gamma*(par.a_grid+par.a_bar)**(-par.Sigma)

    for i_fix in nb.prange(par.Nfix):

        v_a = np.zeros((par.Nz, par.Na))

        for i_z in range(par.Nz):

            # b. cash-on-hand with capital gains on equity exposure, newborns get a bequest instead of their own wealth
            newborn = par.newborn_grid[i_z]
            y = (1-par.tau_l)*w*par.Z_grid[i_fix]*par.e_grid[i_z]
            wealth = (1+rB)*par.a_grid + cg*par.exposure_grid
            m = (1-newborn)*wealth + newborn*bequest*par.Z_grid[i_fix] + y

            # c. savings
            if ss:

                a[i_fix, i_z, :] = 0.0

            else:

                c_endo = np.nan*par.a_grid  # write your code here, don't forget the taste for wealth!
                m_endo = np.nan*par.a_grid  # write your code here
                interp_1d_vec(m_endo, par.a_grid, m, a[i_fix, i_z])
                a[i_fix, i_z, :] = np.fmax(a[i_fix, i_z, :], 0.0)  # borrowing constraint
                a[i_fix, i_z, :] = np.fmin(a[i_fix, i_z, :], par.a_grid[-1])  # top of grid

            c[i_fix, i_z] = m-a[i_fix, i_z]

            # d. envelope condition
            v_a[i_z] = (1-newborn)*(1+rB)*c[i_fix, i_z]**(-par.sigma)

        # e. expectation step
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

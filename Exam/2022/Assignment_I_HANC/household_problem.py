import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,wt,rt,transfer,vbeg_a_plus,vbeg_a,a,c,ell,l,inc,u):
    """ solve backwards with vbeg_a_plus from previous iteration """

    for i_fix in range(par.Nfix):
        
        zeta = par.zeta_grid[i_fix]
        varphi = par.varphi_grid[i_fix]

        # a. solution step
        for i_z in range(par.Nz):

            # i. prepare
            z = par.z_grid[i_z]
            fac = (wt*zeta*z/varphi)**(1/par.nu)

            # ii. use focs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma)
            ell_endo = fac*c_endo**(-par.sigma/par.nu)
            l_endo = ell_endo*zeta*z

            # iii. re-interpolate
            m_endo = c_endo + par.a_grid - wt*l_endo - transfer*(i_z == 0)
            m_exo = (1+rt)*par.a_grid

            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo,m_exo,ell[i_fix,i_z,:])
            l[i_fix,i_z,:] = ell[i_fix,i_z,:]*zeta*z

            # iv. saving
            a[i_fix,i_z,:] = m_exo + wt*l[i_fix,i_z,:] - c[i_fix,i_z,:]

            # v. refinement at constraint
            for i_a in range(par.Na):

                if a[i_fix,i_z,i_a] < 1e-8:
                    
                    # o. binding constraint for a
                    a[i_fix,i_z,i_a] = 0.0

                    # oo. solve foc for n
                    elli = ell[i_fix,i_z,i_a]

                    it = 0
                    while True:

                        ci = (1+rt)*par.a_grid[i_a] + wt*elli*zeta*z + transfer*(i_z == 0)

                        error = elli - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < par.tol_ell:
                            break
                        else:
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*wt*zeta*z
                            elli = elli - error/derror

                        it += 1
                        if it > par.max_iter_ell: raise ValueError('too many iterations searching for ell')

                    # ooo. save
                    c[i_fix,i_z,i_a] = ci
                    ell[i_fix,i_z,i_a] = elli
                    l[i_fix,i_z,i_a] = elli*zeta*z
                    
                else:

                    break

        inc[i_fix] = wt*l[i_fix] + rt*par.a_grid + transfer*(i_z == 0)
        u[i_fix,:,:] = c[i_fix]**(1-par.sigma)/(1-par.sigma) - varphi*ell[i_fix]**(1+par.nu)/(1+par.nu)

        # b. expectation step
        v_a = c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = (1+rt)*z_trans[i_fix]@v_a
# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit        
def solve_hh_backwards(par,z_trans,w,r,d,tau,vbeg_a_plus,vbeg_a,a,c,ell,n,ss=False):
    """ solve backwards with vbeg_a_plus from previous iteration """

    for i_fix in range(par.Nfix):
        
        # a. solution step
        for i_z in range(par.Nz):

            # i. prepare
            z = par.z_grid[i_z]
            T = d*z - tau*z
            fac = (w*z/par.varphi)**(1/par.nu)

            if ss:

                ell[i_fix,i_z,:] = 1.0
                n[i_fix,i_z,:] = ell[i_fix,i_z,:]*z
                a[i_fix,i_z,:] = 0.0
                c[i_fix,i_z,:] = (1+r)*par.a_grid + w*n[i_fix,i_z,:] + T
                
                continue

            # ii. use focs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma)
            ell_endo = fac*(par.beta*vbeg_a_plus[i_fix,i_z,:])**(1/par.nu)
            n_endo = ell_endo*z

            # iii. re-interpolate
            a_lag_endo = (c_endo + par.a_grid - w*n_endo - T)/(1+r)
            a_lag = par.a_grid

            interp_1d_vec(a_lag_endo,c_endo,a_lag,c[i_fix,i_z,:])
            interp_1d_vec(a_lag_endo,ell_endo,a_lag,ell[i_fix,i_z,:])

            n[i_fix,i_z,:] = ell[i_fix,i_z,:]*z

            # iv. saving
            a[i_fix,i_z,:] = (1+r)*a_lag + w*n[i_fix,i_z,:] + T - c[i_fix,i_z,:]

            # v. refinement at constraint
            for i_a in range(par.Na):

                if a[i_fix,i_z,i_a] < par.a_min:
                    
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a] = par.a_min

                    # ii. solve foc for n
                    elli = ell[i_fix,i_z,i_a]
                    for i in range(30):

                        ci = (1+r)*par.a_grid[i_a] + w*z*elli + T - par.a_min # from binding constraint

                        error = elli - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < 1e-11:
                            break
                        else:
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*w*z
                            elli = elli - error/derror
                    else:
                        
                        raise ValueError('solution could not be found')

                    # iii. save
                    c[i_fix,i_z,i_a] = ci
                    ell[i_fix,i_z,i_a] = elli
                    n[i_fix,i_z,i_a] = elli*z
                    
                else:

                    break

        # b. expectation step
        v_a = c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = (1+r)*z_trans[i_fix]@v_a
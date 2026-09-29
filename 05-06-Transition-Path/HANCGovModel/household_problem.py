import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,pB,tau,vbeg_a_plus,vbeg_a,a,c,u):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
        
            # i. EGM
            c_endo = (par.beta/pB*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + pB*par.a_grid
            
            # ii. interpolation to fixed grid
            m = par.a_grid + (1-tau)*par.z_grid[i_z]
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-pB*a[i_fix,i_z]
            u[i_fix,i_z] = c[i_fix,i_z]**(1-par.sigma)/(1-par.sigma)

        # b. expectation step
        v_a = c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a
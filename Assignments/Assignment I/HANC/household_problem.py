import numpy as np
import numba as nb
from consav.linear_interp import interp_1d_vec

@nb.njit
def u(c,ell, sigma, frisch, vphi):
    return (c**(1-sigma) - 1) / ((1-sigma)) - vphi*ell**(1+1/frisch) / (1+1/frisch)


@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,l,ell,
                       tau_l,tau_a,taxes,transfer,ss=False):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # par,z_trans: always inputs
    # r,w,tau_l,tau_a: inputs because they are in .inputs_hh
    # vbeg_a, vbeg_a_plus: input because vbeg_a is in .intertemps_hh
    # a,c,l: outputs because they are in .outputs_hh

    # ss = True is to get guess of vbeg_a

    for i_fix in nb.prange(par.Nfix): # fixed types

        # a. solve step
        for i_z in nb.prange(par.Nz): # stochastic discrete states

            ## i. labor supply and Euler inversion 
            we = w*par.z_grid[i_z]*(1-tau_l) # after tax effective wage 
            muc_nextgrid = par.beta * vbeg_a_plus[i_fix,i_z]
            c_endo = muc_nextgrid**(-1/par.sigma)
            ell_nextgrid = (we * muc_nextgrid / par.vphi) ** par.frisch

            ## ii. cash-on-hand
            m_endo = c_endo -  we*ell_nextgrid  + par.a_grid - transfer
            m =  (1+(1-tau_a)*r)*par.a_grid

            if ss: # initial values 

                a[i_fix,i_z,:] = 0.05

            else:

                # interpolation to fixed grid
                interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
                interp_1d_vec(m_endo,ell_nextgrid,m,ell[i_fix,i_z])
            
                # if constrained we have to solve labor supply decision again 
                a_min = par.a_grid[0]
                for i_a in range(par.Na):
                    if a[i_fix,i_z,i_a] < a_min: # constrained 
                        a[i_fix,i_z,i_a] = a_min
                        other_income = (1+(1-tau_a)*r)*par.a_grid[i_a] + transfer
                        ell[i_fix,i_z,i_a] = solve_cl(we, other_income, a_min, par.vphi, par.sigma, par.frisch)

            # cash-on-hand 
            m = (1+(1-tau_a)*r)*par.a_grid + we*l[i_fix,i_z,:] + transfer

            c[i_fix,i_z] = m - a[i_fix,i_z]
            l[i_fix,i_z] = l[i_fix,i_z]*par.z_grid[i_z]
            taxes[i_fix,i_z] = tau_a*r*par.a_grid + l[i_fix,i_z]*w*tau_l

        # b. expectation step 
        v_a = (1+(1-tau_a)*r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a
        

@nb.njit 
def muc(c, sigma):
    return c**(-sigma)

@nb.njit 
def inv_mld(mld, vphi, frisch):
    return  (mld/vphi)**frisch


@nb.njit 
def solve_cl(we, other_income, a_min, vphi, sigma, frisch, damp=0.5, N=200):

    # initial guess is analytical solution when other_income = 0
    l_guess = (we**(1-sigma) /vphi)**(1/(1/frisch + sigma))

    for i in range(N):
        # get c from budget constraint 
        c = l_guess*we + other_income - a_min

        # get updated l from foc 
        mld = we*muc(c,sigma)
        l_new = inv_mld(mld, vphi, frisch)

        # labor supply must be positive 
        l_new = np.fmax(l_new, 1e-04)
        
        if np.abs(l_new-l_guess) < 1E-09: 
            break 
        else:
            l_guess = l_guess + damp*(l_new - l_guess)
    
    else:

        raise ValueError(f"Cannot solve constrained household's problem: No convergence after {N} iterations!")
    
    return l_new



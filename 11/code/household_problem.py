import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,
    shock_beta,w,RealR_ex_post,tau,u_bar,phi_obar,transfer,
    vbeg_a_plus,vbeg_a,a,c,y,u_UI,ss=False):
    """ solve backwards with vbeg_a_plus from previous iteration """

    for i_fix in range(par.Nfix):

        # a. solution step
        for i_z in range(par.Nz):
      
            z = par.z_grid[i_z] # productivity
            i_u = par.i_u_hh[i_fix,i_z] # unemployment indicator

            # i. income
            if i_u == 0:
                u_UI_ = 0.0
                yt = w*z
            else:

                u_UI_ = np.fmax(np.fmin(u_bar-(i_u-1),1.0),0.0)
                
                # if i_u < u_bar-2:
                #     u_UI_ = 1.0
                # elif i_u > u_bar+2:
                #     u_UI_ = 0.0
                # else:
                #     gamma = 5
                #     u_UI_ = np.exp(gamma*(u_bar+0.5-i_u))/(1+np.exp(gamma*(u_bar+0.50-i_u)))

                yt = (u_UI_*phi_obar + (1-u_UI_)*par.phi_ubar)*w*z

            u_UI[i_fix,i_z,:] = u_UI_

            # ii. income after tax
            y[i_fix,i_z,:] = (1-tau)*yt + transfer

            # iii. EGM
            m = RealR_ex_post*par.a_grid + y[i_fix,i_z,:]

            # iv. consumption-saving
            if par.beta_grid[i_fix] < par.beta_HtM: # HtM
        
                a[i_fix,i_z,:] = 0.0
                c[i_fix,i_z,:] = m

            elif ss:

                c[i_fix,i_z,:] = 0.9*m
                a[i_fix,i_z,:] = m-c[i_fix,i_z,:]

            else:

                # o. EGM
                c_endo = (shock_beta*par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
                m_endo = c_endo + par.a_grid
            
                # oo. interpolation to fixed grid
                interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])

                # ooo. enforce borrowing constraint
                a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0)

                # oooo. implied consumption
                c[i_fix,i_z] = m-a[i_fix,i_z]

        # b. expectation step
        v_a = RealR_ex_post*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

################
# fill_z_trans #
################

import math
inv_sqrt2 = 1/math.sqrt(2) # precompute

@nb.njit(fastmath=True)
def _norm_cdf(z):  
    """ raw normal cdf """

    return 0.5*math.erfc(-z*inv_sqrt2)

@nb.njit(fastmath=True)
def norm_cdf(z,mean,std):
    """ normal cdf with scaling """

    # a. check
    if std <= 0:
        if z > mean: return 1
        else: return 0

    # b. scale
    z_scaled = (z-mean)/std

    # c. return
    return _norm_cdf(z_scaled)

@nb.njit(fastmath=True)
def fill_z_trans(par,z_trans,delta,lambda_u):
    """ transition matrix for z """
    
    # a. logarithm
    log_e_grid = np.log(par.e_grid)

    # b. transition matrix
    for i_fix in nb.prange(par.Nfix):
        for i_e in nb.prange(par.Ne):
            for i_u in nb.prange(par.Nu+1):
                for i_e_plus in nb.prange(par.Ne):
                    for i_u_plus in nb.prange(par.Nu+1):

                        i_z = i_u*par.Ne + i_e
                        i_z_plus = i_u_plus*par.Ne+i_e_plus

                        # i. u_trans
                        if i_u == 0:
                            
                            if i_u_plus == 0:
                                u_trans = 1.0-delta*(1-lambda_u)
                            elif i_u_plus == 1:
                                u_trans = delta*(1-lambda_u)*par.UI_prob
                            elif i_u_plus == par.Nu:
                                u_trans = delta*(1-lambda_u)*(1-par.UI_prob)
                            else:
                                u_trans = 0.0
                        
                        else:

                            if i_u_plus == 0:
                                u_trans = lambda_u
                            elif (i_u_plus == i_u+1) or (i_u_plus == i_u == par.Nu):
                                u_trans = 1.0-lambda_u
                            else:
                                u_trans = 0.0

                        z_trans[i_fix,i_z,i_z_plus] = u_trans

                        # b. e_trans
                        if par.Ne > 1:

                            if i_e_plus == par.Ne-1:
                                L = 1.0
                            else:
                                midpoint = log_e_grid[i_e_plus] + (log_e_grid[i_e_plus+1]-log_e_grid[i_e_plus])/2
                                L = norm_cdf(midpoint,par.rho_e*log_e_grid[i_e],par.sigma_psi)

                            if i_e_plus == 0:
                                R = 0.0
                            else:
                                midpoint = log_e_grid[i_e_plus] - (log_e_grid[i_e_plus]-log_e_grid[i_e_plus-1])/2
                                R = norm_cdf(midpoint,par.rho_e*log_e_grid[i_e],par.sigma_psi)

                            z_trans[i_fix,i_z,i_z_plus] *= (L-R)      
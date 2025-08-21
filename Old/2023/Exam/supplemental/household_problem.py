import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,
    delta,lambda_u_s,w,r,tau,div,transfer,
    vbeg_a_plus,vbeg_a,a,c,u_ALL,u_UI,ss=False):

    s = np.zeros_like(a)
    
    # a. solution step
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            
            i_u = par.i_u_hh[i_fix,i_z] # unemployment indicator

            # i. income
            if i_u == 0:
                u_UI_ = 0.0
                yt = w
                u_ALL[i_fix,i_z,:] = 0.0
            else:
                u_UI_ = np.fmax(np.fmin(par.u_bar_ss-(i_u-1),1.0),0.0)
                yt = (u_UI_*par.phi_obar + (1-u_UI_)*par.phi_ubar)*w
                u_ALL[i_fix,i_z,:] = 1.0

            u_UI[i_fix,i_z,:] = u_UI_

            # ii. income after tax
            y = (1-tau)*yt + div + transfer

            # iii. EGM
            m = (1+r)*par.a_grid + y

            # iv. consumption-saving
            if i_fix == 0:
        
                a[i_fix,i_z,:] = 0.0
                c[i_fix,i_z,:] = m

            elif ss:

                c[i_fix,i_z,:] = 0.9*m
                a[i_fix,i_z,:] = m-c[i_fix,i_z,:]

            else:

                # o. EGM
                c_endo = (par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
                m_endo = c_endo + par.a_grid

                # oo. interpolation to fixed grid
                interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])

                # ooo. enforce borrowing constraint
                a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0)

                # oooo. implied consumption and value
                c[i_fix,i_z] = m-a[i_fix,i_z]

    # b. update transition matrix
    fill_s(par,s)
    fill_z_trans(par,z_trans,delta,lambda_u_s,s)

    # c. expectation step
    v_a = (1+r)*c**(-par.sigma)
    
    for i_fix in range(par.Nfix):
        for i_z_lag in range(par.Nz):
            
            vbeg_a[i_fix,i_z_lag,:] = 0.0
            for i_z in range(par.Nz):
                vbeg_a[i_fix,i_z_lag,:] += z_trans[i_fix,:,i_z_lag,i_z]*v_a[i_fix,i_z,:]

#####################
# transition matrix #
#####################

@nb.njit
def fill_s(par,s):
    """ fill search intensity """

    for i_fix in range(par.Nfix):

        s[i_fix,0,:] = 0.0
        for i_z_lag in range(1,par.Nz): 
            s[i_fix,i_z_lag,:] = 1.0

@nb.njit
def fill_z_trans(par,z_trans,delta,lambda_u_s,s):
    """ transition matrix for z """
    
    for i_fix in range(par.Nfix):
        for i_a in range(par.Na):
            for i_z_lag in range(par.Nz):
                for i_z in range(par.Nz):
                            
                    i_u_lag = par.i_u_hh[i_fix,i_z_lag] 
                    i_u = par.i_u_hh[i_fix,i_z] 

                    if i_u_lag == 0: # working last period
                        
                        if i_u == 0:
                            z_trans_ = 1.0-delta
                        elif i_u == 1:
                            z_trans_ = delta
                        else:
                            z_trans_ = 0.0
                    
                    else: # unemployed last

                        if i_u == 0:
                            z_trans_ = lambda_u_s*s[i_fix,i_z_lag,i_a]
                        elif (i_u == i_u_lag+1) or (i_u == i_u_lag == par.Nu):
                            z_trans_ = 1.0-lambda_u_s*s[i_fix,i_z_lag,i_a]
                        else:
                            z_trans_ = 0.0

                    z_trans_ = np.fmin(z_trans_,1.0)
                    z_trans_ = np.fmax(z_trans_,0.0)

                    z_trans[i_fix,i_a,i_z_lag,i_z] = z_trans_
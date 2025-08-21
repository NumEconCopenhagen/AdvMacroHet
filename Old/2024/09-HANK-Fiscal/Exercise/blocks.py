import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def production(par,ini,ss,pi_w,L,w,pi,Y):

    w[:] = par.Gamma
    Y[:] = par.Gamma*L
    pi[:] = pi_w 
    

@nb.njit
def central_bank(par,ini,ss,pi,r):

    r[:] = ss.r + (par.phi_pi-1.)*pi
        

@nb.njit
def government(par,ini,ss,G,chi,B,r,tau,Y,taxes,Z,w,L):

    chi[:] = ss.chi 

    for t in range(par.T):
        
        B_lag = B[t-1] if t > 0 else ini.B

        if par.omega > 5.: 

            B[t] = ss.B 
            tau[t] = ((1+r[t])*B_lag + G[t] + chi[t] - B[t])/Y[t]

            taxes[t] = tau[t]*Y[t]

        else:

            tau[t] = ss.tau + par.omega*(B_lag-ss.B)/ss.Y
            taxes[t] = tau[t]*Y[t]
            B[t] = ((1+r[t])*B_lag + G[t] + chi[t] - taxes[t]) 

    # post-tax income 
    Z[:] = (1-tau)*w*L


@nb.njit
def NKWC(par,ini,ss,pi_w,L,w,C_hh,NKWC_res):

    # a. phillips curve
    pi_w_plus = lead(pi_w,ss.pi_w)

    LHS = pi_w
    RHS = par.kappa*(par.varphi*L**par.nu - 1/par.mu*w*C_hh**(-par.sigma)) + par.beta*pi_w_plus
    NKWC_res[:] = LHS-RHS

@nb.njit
def market_clearing(par,ini,ss,G,B,Y,C_hh,A_hh,A,clearing_A,clearing_Y):
        
    # a. aggregates
    A[:] = B

    # b. market clearing
    clearing_A[:] = A-A_hh
    clearing_Y[:] = Y-C_hh-G
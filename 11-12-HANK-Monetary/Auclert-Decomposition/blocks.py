import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def production(par,ini,ss,pi_w,L,w,pi,Y):

    Y[:] = L
    w[:] = 1/par.mu
    pi[:] = pi_w
    

@nb.njit
def central_bank(par,ini,ss,pi,r,i):

    pi_plus = lead(pi,ss.pi)

    if par.taylor:
        i[:] = ss.i + par.phi_pi*pi
        r[:] = (1+i)/(1+pi_plus)-1
    else:
        r[:] = ss.r
        i[:] = (1+r)*(1+pi_plus)-1
        

@nb.njit
def government(par,ini,ss,G,pi,i,r,Y,B,q,rb,tau,taxes):

    # a. price of a bond paying 1, delta_B, delta_B**2, ... from t+1 on, in nominal terms if bonds are nominal
    for k in range(par.T):

        t = par.T-1-k
        q_plus = q[t+1] if t < par.T-1 else ss.q
        if par.nominal_bonds:
            q[t] = (1+par.delta_B*q_plus)/(1+i[t])
        else:
            q[t] = (1+par.delta_B*q_plus)/(1+r[t])

    for t in range(par.T):
        
        B_lag = B[t-1] if t > 0 else ini.B
        q_lag = q[t-1] if t > 0 else ss.q

        # b. realized real return on bonds held from t-1 to t: coupon plus the value of the rest
        if par.nominal_bonds:
            rb[t] = (1+par.delta_B*q[t])/q_lag/(1+pi[t])-1
        else:
            rb[t] = (1+par.delta_B*q[t])/q_lag-1

        # c. debt path
        B[t] = ss.B + par.rho_B*(B_lag-ss.B+G[t]-ss.G)

        # d. taxes adjust
        taxes[t] = (1+rb[t])*B_lag + G[t] - B[t]
        tau[t] = taxes[t]/Y[t]


@nb.njit
def mutual_fund(par,ini,ss,r,rb,tau,Y,L,w,B,Z,d,p,A,ra):

    # a. post-tax labor income and dividends
    Z[:] = (1-tau)*w*L
    d[:] = (1-tau)*(1-1/par.mu)*Y

    # b. stock price from no-arbitrage
    for k in range(par.T):

        t = par.T-1-k
        p_plus = p[t+1] if t < par.T-1 else ss.p
        d_plus = d[t+1] if t < par.T-1 else ss.d
        p[t] = (p_plus+d_plus)/(1+r[t])

    # c. assets and return
    A[:] = p + B

    for t in range(par.T):
        if t == 0:
            ra[t] = (p[t]+d[t]+(1+rb[t])*ss.B)/ss.A-1
        else:
            ra[t] = r[t-1]


@nb.njit
def NKWC(par,ini,ss,pi_w,L,tau,w,C_hh,NKWC_res):

    pi_w_plus = lead(pi_w,ss.pi_w)

    LHS = pi_w
    RHS = par.kappa*(par.varphi*L**par.nu - (1-tau)*w*C_hh**(-par.sigma)) + par.beta_avg*pi_w_plus
    NKWC_res[:] = LHS-RHS


@nb.njit
def market_clearing(par,ini,ss,G,A,A_hh,Y,C_hh,clearing_A,clearing_Y):
        
    clearing_A[:] = A-A_hh
    clearing_Y[:] = Y-C_hh-G

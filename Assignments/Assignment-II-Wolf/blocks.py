import numpy as np
import numba as nb

from GEModelTools import lag, lead


@nb.njit
def production(par,ini,ss,pi_w,L,w,pi,Y):

    w[:] = ss.w
    Y[:] = L
    pi[:] = pi_w


@nb.njit
def central_bank(par,ini,ss,eps_r,r,ra):

    # a. real-rate rule
    r[:] = ss.r + eps_r

    # b. bonds pay the rate set in the previous period
    ra[:] = lag(ss.r,r)


@nb.njit
def government(par,ini,ss,G,chi,ra,Y,w,L,tau,taxes,B,Z):

    # a. taxes cover spending and the interest on steady-state debt
    taxes[:] = G + ra*ss.B
    tau[:] = taxes/Y

    # b. debt pays for the transfers
    for t in range(par.T):

        B_lag = B[t-1] if t > 0 else ini.B
        B[t] = (1+ra[t])*B_lag + G[t] + chi[t] - taxes[t]

    # c. post-tax income
    Z[:] = (1-tau)*w*L


@nb.njit
def NKWC(par,ini,ss,pi_w,L,w,C_hh,NKWC_res):

    pi_w_plus = lead(pi_w,ss.pi_w)

    LHS = pi_w
    RHS = par.kappa*(par.varphi*L**par.nu - (1-ss.tau)/par.mu_w*w*C_hh**(-par.sigma)) + par.beta*pi_w_plus
    NKWC_res[:] = LHS-RHS


@nb.njit
def market_clearing(par,ini,ss,G,B,Y,C_hh,A_hh,clearing_A,clearing_Y):

    clearing_A[:] = B-A_hh
    clearing_Y[:] = Y-C_hh-G

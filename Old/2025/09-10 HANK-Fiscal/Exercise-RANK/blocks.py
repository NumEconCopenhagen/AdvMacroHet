
import numpy as np
import numba as nb
from GEModelTools import lag, lead

# @nb.njit(cache=False)
def NK_block(par,ini,ss,i,pi,P,C,Y,Euler,r,Z,N,w,mc,goods_mkt,NKPC,beta,eps_i,Labor_supply,profits,G, LT,A,B):

    # Inflation 
    pi[:] = P/lag(ss.P,P) - 1.
    
    # Monetary policy 
    if par.ZLB:
        ...
    else:
        i[:] = ss.i + par.phi * pi + eps_i 

    pi_p = lead(pi, ss.pi)
    r[:] = (1 + i) / (1 + pi_p) - 1

    # Euler 
    C_p = lead(C, ss.C)
    Euler[:] = C**(-par.CRRA) - (1 + r) * beta * C_p**(-par.CRRA)

    # Production 
    N[:] = Y / Z

    # Firms / NKPC
    mc[:] = w / Z   
    NKPC[:] = pi*(1+pi) - par.kappa * (mc - 1/par.mu) + par.betaF * pi_p*(1+pi_p)
    profits[:] = Y - (w*N + par.theta/2 * pi**2 * Y)

    # Labor supply 
    Labor_supply[:] = par.vphi * N ** (par.inv_frisch) - w * C ** (-par.CRRA)

    # Market clearing 
    goods_mkt[:] = Y - (C + G + par.theta/2 * pi**2 * Y)

    # Government bonds 
    i_lag = lag(ss.i,i)
    B[:] = ss.B 
    LT[:] = (1+i_lag) / (1+pi) * ss.B + G - ss.B

    # HH budget
    for t in range(par.T):
        if t==0:
            A_lag = ss.A
            i_lag = ss.i 
        else:
            A_lag = A[t-1]
            i_lag = i[t-1]
        A[t] = (w*N + profits - LT)[t] - C[t] + (1+i_lag)/(1+pi[t])*A_lag



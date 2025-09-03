import numpy as np
import numba as nb

from GEModelTools import prev,next

@nb.njit
def government(par,ini,ss,tau,G,pB,B):

    for t in range(par.T):
        
        B_lag = prev(B,t,ini.B)
        tau[t] = ss.tau + par.phi*(B_lag-ss.B)
        B[t] = (B_lag + G[t] - tau[t])/pB[t]

@nb.njit
def market_clearing(par,ini,ss,B,A_hh,clearing_B):

    clearing_B[:] = B-A_hh
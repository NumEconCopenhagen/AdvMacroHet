import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit 
def production_firm(par,ini,ss,Gamma,K,rK,w,Y,I):

    alpha = par.alpha
    delta = par.delta
    K_lag = lag(ini.K,K)

    Y[:] = Gamma*K_lag**alpha
    rK[:] = alpha*Y/K_lag
    w[:] = (1.0-alpha)*Y
    I[:] = K - (1-delta)*K_lag

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K

    # b. return
    r[:] = rK - par.delta

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,Y,C_hh,Z_hh,I,clearing_A,clearing_Y,clearing_Z):

    clearing_A[:] = A-A_hh
    clearing_Y[:] = Y-C_hh-I
    clearing_Z[:] = Z_hh-1.0
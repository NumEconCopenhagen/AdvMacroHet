import numpy as np
import numba as nb

from GEModelTools import lag

@nb.njit
def production_firm(par,ini,ss,Gamma,K,L,rK,w,Y):

    K_lag = lag(ini.K,K)

    rK[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
    w[:] = (1.0-par.alpha)*Gamma*(K_lag/L)**par.alpha
    Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    A[:] = K
    r[:] = rK-par.delta

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L,Y,C_hh,K,I,clearing_A,clearing_L,clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_L[:] = L-1.0
    I[:] = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I

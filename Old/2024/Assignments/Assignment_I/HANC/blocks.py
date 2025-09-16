import numpy as np
import numba as nb

from GEModelTools import lag, lead


@nb.njit # required decorator for numba
def production_firm(par,ini,ss,Gamma,K,L,rK,w,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
    w[:] = (1.0-par.alpha)*Gamma*(K_lag/L)**par.alpha
    
    # b. production and investment
    Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)

@nb.njit
def transfers(par,ini,ss,transfer):
    transfer[:] = ss.transfer

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K

    # b. return
    r[:] = rK-par.delta

@nb.njit
def market_clearing(par,ini,ss,
                    A,A_hh,L,L_hh,K,Y,C_hh,TAXES_hh, # inputs
                    I,clearing_A,clearing_L,clearing_Y, transfer, clearing_G # outputs
                    ):

    clearing_A[:] = A-A_hh
    clearing_L[:] = L-L_hh
    I[:] = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I
    clearing_G[:] = TAXES_hh - transfer 
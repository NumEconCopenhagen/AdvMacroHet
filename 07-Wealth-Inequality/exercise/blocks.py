import numpy as np
import numba as nb

from GEModelTools import lag, lead

# lags and leads of unknowns and shocks
# K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
# K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]

# @nb.njit
# def lag(inivalue,pathvalue):

#     output = np.empty_like(pathvalue)
#     output[0,:] = inivalue
#     output[1:,:] = pathvalue[:-1,:]
#     return output

# @nb.njit
# def lead(pathvalue,ssvalue):

#     output = np.empty_like(pathvalue)
#     output[:-1,:] = pathvalue[1:,:]
#     output[-1,:] = ssvalue
#     return output

@nb.njit # required decorator for numba
def production_firm(par,ini,ss,Gamma,K,L,rK,w,Y,tau_K,tau_r):

    # par: parameters
    # ini: initial state (e.g. ss)
    # ss: steady state (at end)
    # *: input (Gamma,K,L) and output (rk,w,Y) variables (order does not matter)

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
    w[:] = (1.0-par.alpha)*Gamma*(K_lag/L)**par.alpha
    
    # b. production and investment
    Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r,tau_K,tau_r):

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

    # expenses to transfer
    clearing_G[:] = TAXES_hh - transfer 
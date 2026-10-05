import numpy as np
import numba as nb

from GEModelTools import lag, next


@nb.njit
def production_firm(par, ini, ss, K, Y, w, r, I, div):

    K_lag = lag(ini.K, K)

    # a. production and wages
    Y[:] = np.nan  # write your code here
    w[:] = np.nan  # write your code here

    # b. investment FOC
    r[:] = np.nan  # write your code here

    # c. dividends
    I[:] = np.nan  # write your code here
    div[:] = np.nan  # write your code here


@nb.njit
def mutual_fund(par, ini, ss, r, div, q, rB, cg, ra, A):

    # a. firm value from no-arbitrage with dividend taxes (backwards in time)
    for t_ in range(par.T):
        t = (par.T-1)-t_  # backwards in time (numba does not support reversed)
        q_plus = np.nan  # write your code here
        div_plus = np.nan  # write your code here
        r_plus = np.nan  # write your code here
        q[t] = np.nan  # write your code here

    # b. bonds pay the return expected when they were bought
    rB[0] = ini.r
    rB[1:] = r[1:]

    # c. the unexpected capital gain in the first period is paid per unit of equity exposure, int s(a)a dD
    A[:] = q+par.B
    equity_payoff = (1-par.tau_d)*div[0]+q[0]
    exposure = np.sum(ini.Dbeg*par.exposure_grid)
    cg[0] = (equity_payoff-(1+rB[0])*ini.q)/exposure
    cg[1:] = 0.0

    # d. average ex-post return on wealth
    ra[0] = (equity_payoff+(1+rB[0])*par.B)/(ini.q+par.B)-1
    ra[1:] = r[1:]


@nb.njit
def bequests(par, ini, ss, q, ra, bequest):

    # the dead hold a share xi of wealth, newborns receive it net of the estate tax in proportion to Z
    A_lag = lag(ini.q, q)+par.B
    bequest[:] = (1-par.tau_E)*(1+ra)*A_lag


@nb.njit
def government(par, ini, ss, w, div, q, ra, rB, G):

    A_lag = lag(ini.q, q)+par.B

    # a. revenue
    labor_tax = par.tau_l*w
    dividend_tax = par.tau_d*div
    estate_tax = par.tau_E*par.xi*(1+ra)*A_lag

    # b. spending adjusts, debt is constant
    G[:] = labor_tax+dividend_tax+estate_tax-rB*par.B


@nb.njit
def market_clearing(par, ini, ss, A, A_hh, Y, C_hh, I, G, clearing_A, clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_Y[:] = Y-C_hh-I-G

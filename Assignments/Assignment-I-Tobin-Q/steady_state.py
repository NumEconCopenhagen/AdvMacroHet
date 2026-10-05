import time
import numpy as np
import pandas as pd
from scipy import optimize

from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import tails


def permanent_income_grid(year):
    """ permanent income levels of the bottom 50%, next 40%, top 10-1%, top 1-0.1% and top 0.1% (PSZ labor income shares) """

    df = pd.read_csv('labor_psz.csv').set_index('year')
    bottom50, next40, top10, top1, top01 = df.loc[year, [
        'bottom50', 'next40', 'top10', 'top100', 'top1000']]/100

    income_shares = np.array([bottom50, next40, top10-top1, top1-top01, top01])
    share_grid = np.array([0.5, 0.4, 0.09, 0.009, 0.001])

    return income_shares/share_grid, share_grid


def wealth_shares_data(year):
    """ PSZ wealth shares """

    df = pd.read_csv('wealth_psz.csv')
    return df[df['Year'] == year].set_index('Quantile')['value']


def prepare_hh_ss(model):
    """ set the grids, the transition matrix and the initial guesses for the household block """

    par = model.par
    ss = model.ss

    # a. equity exposure s(a)a from the portfolio rule (the asset grid is built in allocate)
    if par.portfolio_rule:
        log_relative_wealth = np.log(np.fmax(par.a_grid/par.A_ref, 1e-4))
        equity_share = par.psi0/(1+np.exp(-par.psi1*(log_relative_wealth-par.psi2)))
    else:
        equity_share = np.ones(par.Na)

    par.exposure_grid[:] = equity_share*par.a_grid

    # b. idiosyncratic productivity and death, i_z = 2*i_e + newborn
    e_grid, e_trans, e_ergodic, _, _ = log_rouwenhorst(par.rho_e, par.sigma_e*np.sqrt(1-par.rho_e**2), par.Ne)
    death_trans = np.array([[1-par.xi, par.xi], [1-par.xi, par.xi]])
    death_ergodic = np.array([1-par.xi, par.xi])

    par.e_grid[:] = np.repeat(e_grid, 2)
    par.newborn_grid[:] = np.tile(np.array([0.0, 1.0]), par.Ne)
    z_ergodic = np.kron(e_ergodic, death_ergodic)

    # c. transition matrix and initial distribution (everybody at a_lag = 0)
    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix, :, :] = np.kron(e_trans, death_trans)
        ss.Dbeg[i_fix, :, 0] = par.share_grid[i_fix]*z_ergodic
        ss.Dbeg[i_fix, :, 1:] = 0.0

    # d. initial guess for the intertemporal variables
    model.set_hh_initial_guess()


def blocks_ss(model):
    """ firm, mutual fund, bequests and government in steady state given K """

    par = model.par
    ss = model.ss

    # a. firm
    ss.Y = np.nan  # write your code here
    ss.w = np.nan  # write your code here
    ss.r = np.nan  # write your code here
    ss.I = np.nan  # write your code here
    ss.div = np.nan  # write your code here

    # b. mutual fund and bequests
    ss.q = np.nan  # write your code here
    ss.A = np.nan  # write your code here
    ss.rB = ss.ra = np.nan  # write your code here
    ss.cg = np.nan  # write your code here
    ss.bequest = np.nan  # write your code here

    # c. government
    ss.G = np.nan  # write your code here


def obj_ss(K_ss, model, do_print=False):
    """ asset market clearing error for a guess on steady state capital """

    ss = model.ss

    # a. all blocks except households
    ss.K = K_ss
    blocks_ss(model)

    # b. households: backward step (EGM), then forward step (histogram method)
    model.solve_hh_ss()
    model.simulate_hh_ss()

    # c. market clearing
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I-ss.G

    if do_print:
        print(f'K = {ss.K:8.4f}, r = {ss.r:7.4f}, clearing_A = {ss.clearing_A:12.8f}')

    return ss.clearing_A


def find_ss(model, method='direct', do_print=False):
    """ find the steady state with the direct method or calibrate it """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model, do_print=do_print)
    elif method == 'calibrate':
        find_ss_calibrate(model, do_print=do_print)
    elif method == 'calibrate_beta':
        find_ss_beta(model, do_print=do_print)
    else:
        raise NotImplementedError

    if do_print:
        ss = model.ss
        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.K=:.4f}, {ss.r=:.4f}, {ss.q/ss.K=:.4f}, {ss.A/ss.Y=:.4f}')
        print(f'{ss.clearing_A=:.2e}, {ss.clearing_Y=:.2e}')


def find_ss_direct(model, do_print=False):
    """ find steady state capital with a root-finder """

    par = model.par

    # a. bracket: K such that 0.1% < r < 20%
    K_min = (par.alpha/par.mu*par.Gamma/(0.20+par.delta))**(1/(1-par.alpha))
    K_max = (par.alpha/par.mu*par.Gamma/(0.001+par.delta))**(1/(1-par.alpha))

    # b. search
    optimize.brentq(obj_ss, K_min, K_max, args=(model, do_print), xtol=1e-12)


def find_ss_calibrate(model, do_print=False):
    """ calibrate Gamma and delta in closed form, then beta, gamma and a_bar with a root-finder """

    par = model.par
    ss = model.ss

    # a. targets: Y = 1, r and W/Y (A_ref is the reference wealth in the portfolio rule)
    ss.Y = 1.0
    ss.r = par.r_target
    ss.q = np.nan  # write your code here
    par.A_ref = ss.q+par.B

    # b. closed form: K from Tobin's Q, Gamma from Y, delta from the investment condition
    ss.K = np.nan  # write your code here
    par.Gamma = np.nan  # write your code here
    par.delta = np.nan  # write your code here

    # c. beta, gamma and a_bar clear the asset market and hit the bottom 50% and top 10% wealth shares
    data = wealth_shares_data(par.year)

    def obj(x):
        """ errors in asset market clearing and the two wealth shares for x = (beta,log gamma,log a_bar) """

        errors = np.nan*np.ones(3)  # write your code here
        return errors

    res = None  # write your code here: optimize.root(...)
    assert res.success, res.message
    obj(res.x)


def find_ss_beta(model, do_print=False):
    """ keep all other parameters and find beta such that r equals its target """

    par = model.par
    ss = model.ss

    # a. K from the investment condition at the target r
    ss.K = np.nan  # write your code here

    # b. beta clears the asset market
    # write your code here

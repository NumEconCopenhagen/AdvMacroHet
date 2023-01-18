import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        alpha = path.alpha[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        Gamma = path.Gamma[ncol,:]
        I = path.I[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        r = path.r[ncol,:]
        rK = path.rK[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]

        capital_income = path.capital_income[ncol,:]
        tax = path.tax[ncol,:]
        transfer = path.transfer[ncol,:]
        policy_target = path.policy_target[ncol,:]
        
        #################
        # implied paths #
        #################

        # a. pre-determined and exogenous
        Gamma[:] = ss.Gamma
        K_lag = lag(ini.K,K)
        L[:] = 1.0

        # b. prices and capacity utilization
        rK[:] = alpha*Gamma*(K_lag/L)**(alpha-1.0)
        r[:] = rK-par.delta
        w[:] = (1.0-alpha)*Gamma*(K/L)**alpha

        # c. investment
        I[:] = K-(1-par.delta)*K_lag

        # d. production and consumption
        Y[:] = Gamma*K_lag**(alpha)*L**(1-alpha)

        # e. stocks equal capital
        A[:] = K

        transfer[:] = tax*rK*K_lag
        capital_income[:] = (1-tax)*(rK-par.delta)*K_lag
        policy_target[:] = capital_income - (ss.rK-par.delta)*ss.K

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        alpha = path.alpha[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        Gamma = path.Gamma[ncol,:]
        I = path.I[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        r = path.r[ncol,:]
        rK = path.rK[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        
        std_y = path.std_y[ncol,:]
        std_a = path.std_a[ncol,:]
        skew_a = path.skew_a[ncol,:]

        RK_A_hh = path.RK_A_hh[ncol,:]

        ###########
        # targets #
        ###########

        clearing_A[:] = A-A_hh
        clearing_Y[:] = Y-C_hh-I

        for t in range(par.T):
            std_y[t] = np.sqrt(np.sum(path.Dz[t]*(par.s_grid-1.0)**2))
            std_a[t] = np.sqrt(np.sum(path.D[t]*(path.a[t]-A_hh[t])**2))
            skew_a[t] = np.sum(path.D[t]*((path.a[t]-A_hh[t])/std_a[t])**3)
        
        K_lag = lag(ini.K,K)
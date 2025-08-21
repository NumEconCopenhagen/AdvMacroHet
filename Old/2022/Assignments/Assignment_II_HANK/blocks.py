import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        chi = path.chi[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]

        #################
        # implied paths #
        #################

        Gamma_lag = lag(ini.Gamma,Gamma)

        # a. firms
        w[:] = Gamma
        pi[:] = (1+pi_w)/(Gamma/Gamma_lag)-1
        Y[:] = Gamma*L

        # b. central bank
        for t in range(par.T):
            i_lag = i[t-1] if t > 0 else ini.i
            i[t] = (1+i_lag)**par.rho_i*((1+ss.r)*(1+pi[t])**(par.phi_pi))**(1-par.rho_i)-1

        # c. Fisher
        pi_plus = lead(pi,ss.pi)
        r[:] = (1+i)/(1+pi_plus)-1
        
        # d. government
        for k in range(par.T):
            t = par.T-1-k
            q_plus = q[t+1] if t < par.T-1 else ss.q
            q[t] = (1+par.delta*q_plus)/(1+r[t])
        
        q_lag = lag(ini.q,q)
        ra[:] = (1+par.delta*q)/q_lag-1

        for t in range(par.T):
            
            B_lag = B[t-1] if t > 0 else ini.B
            tau[t] = ss.tau + par.omega*ss.q*(B_lag-ss.B)/ss.Y
            B[t] = ((1+par.delta*q[t])*B_lag + G[t] + chi[t] - tau[t]*Y[t])/q[t]

        # e. aggregates
        A[:] = q*B

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        chi = path.chi[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        
        #################
        # check targets #
        #################

        # a. phillips curve
        pi_w_plus = lead(pi_w,ss.pi_w)

        LHS = pi_w
        RHS = par.kappa*(par.varphi*L**par.nu - 1/par.mu*(1-tau)*w*C_hh**(-par.sigma)) + par.beta*pi_w_plus
        NKWC_res[:] = LHS-RHS

        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_Y[:] = Y-C_hh-G
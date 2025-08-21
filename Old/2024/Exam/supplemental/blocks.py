import numpy as np
import numba as nb
from GEModelTools import lag, lead, next, prev
   
@nb.njit
def production(par,ini,ss,L,w,Y,K,rK,I,Div,K_foc,r, pi, pi_w):

    # a. Production
    mc = 1/par.mu # real marginal cost 
    K_lag = lag(ini.K,K)
    Y[:] = par.Gamma * K_lag ** par.alpha * L ** (1 - par.alpha)
    w[:] = mc * par.Gamma * (1 - par.alpha) * (K_lag ** par.alpha) * (L ** (-par.alpha))
    rK[:] = r + par.delta 

    # b. Investment decision                 
    MPK = mc * par.Gamma * par.alpha * (K_lag ** (par.alpha - 1)) * (L ** (1 - par.alpha))
    K_p = lead(K,ss.K)
    MPK_ss = mc * par.Gamma * par.alpha * (ss.K ** (par.alpha - 1)) * (ss.L ** (1 - par.alpha))
    MPK_p = lead(MPK,MPK_ss)
    dadjK = par.phi_K * (K/K_lag-1)
    dadjK_p = par.phi_K * (K_p/K-1)
    adjK = par.phi_K/2 *  (K_p/K-1)**2 
    
    
    LHS = 1 + dadjK
    RHS = (MPK_p + (1-par.delta) - adjK + dadjK_p * K_p/K) / (1+r)


    if par.phi_K < 20: # Elastic investment
        
        K_foc[:] = LHS - RHS
        
    else: # Inelastic investment (large adjustment cost -> optimal to not adjust capital)
        
        K_foc[:] = K - ss.K 


    # c. Investment and profits
    I[:] = K - (1-par.delta)*K_lag
    Div[:] = Y - w*L - I - par.phi_K/2*(K/K_lag-1)**2 * K_lag
    
    # d. Inflation 
    w_lag = lag(ini.w,w)
    pi_w[:] = (1+pi)*(w/w_lag) -1.


@nb.njit
def central_bank(par,ini,ss,pi,i,r,eps_i):

    # a. Taylor rule 
    i[:] = ss.i + par.phi_pi*pi + eps_i
 
    # b. Fisher
    pi_plus = lead(pi,ss.pi)
    r[:] = (1+i)/(1+pi_plus)-1


        
@nb.njit
def mutual_fund(par,ini,ss,r,ra,Div,pD,pi):

    for t_ in range(par.T):

        t = (par.T-1) - t_

        p_eq_plus = next(pD,t,ss.pD)
        Div_plus = next(Div,t,ss.Div)
        pD[t] = (Div_plus+p_eq_plus) / (1+r[t])

    term_pD = pD[0]+Div[0]
    term_B = (1+ss.i)/(1+pi[0])*ini.B
    
    ra[0] = (term_B+term_pD)/ini.A_hh -1
    ra[1:] = r[:-1]

@nb.njit
def government(par,ini,ss,G,chi,B,r,i,pi,Taxes,Z,w,L, tau):
    
    # a. interest rates on debt  
    i_lag = lag(ini.i,i)
    rB = (1+i_lag)/(1+pi) - 1 
    
    # b. Lumpsum transfer 
    chi[:] = 0. 

    # c. Government budget 
    for t in range(par.T):
        
        B_lag = B[t-1] if t > 0 else ini.B

        Taxes[t] = ss.Taxes + par.omega*((B_lag-ss.B) + (G[t] - ss.G))/ss.Y
        B[t] = (1+rB[t])*B_lag + G[t] - chi[t] - Taxes[t]

    # d. tax rate and post-tax income
    tau[:] = Taxes/(w*L)
    Z[:] = (1-tau)*w*L

@nb.njit
def NKWC(par,ini,ss,pi_w,L,w,C_hh,NKWC_res,tau):

    # a. phillips curve
    pi_w_plus = lead(pi_w,ss.pi_w)

    LHS = pi_w
    RHS = par.kappa*(par.varphi*L**par.nu - (1-tau)/par.mu*w*C_hh**(-par.sigma)) + par.beta*pi_w_plus
    NKWC_res[:] = LHS-RHS

@nb.njit
def market_clearing(par,ini,ss,G,B,Y,C_hh,A_hh,clearing_A,clearing_Y,pD,I,K):

    K_lag = lag(ini.K,K)

    # a. market clearing
    clearing_A[:] = pD + B - A_hh
    clearing_Y[:] = Y - C_hh - G - I  - par.phi_K/2*(K/K_lag-1)**2 * K_lag 
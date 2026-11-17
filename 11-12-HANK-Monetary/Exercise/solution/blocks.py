import numpy as np
import numba as nb
from GEModelTools import lag, lead, next, prev
   
@nb.njit
def production(par,ini,ss,L,w,Y,Div,Z):

    # a. production
    Y[:] = par.Gamma * L
    w[:] =  par.Gamma / par.mu
    Z[:] =w*L
    Div[:] = Y - Z 


@nb.njit
def central_bank(par,ini,ss,pi_w,pi,i,r,eps_i):

    # constant r rule 
    # r[:] = ss.i + eps_i
    # i[:] = r + pi 
    # pi[:] = pi_w 

    # Taylor rule 
    pi[:] = pi_w 
    i[:] = ss.i + par.phi_pi*pi + eps_i
    # # c. Fisher
    pi_plus = lead(pi,ss.pi)
    r[:] = (1+i)/(1+pi_plus)-1


@nb.njit
def government(par,ini,ss,chi,B,pi,i):
    i_lag = lag(ini.i,i)
    rb = (1+i_lag)/(1+pi) - 1

    for t in range(par.T):
        
        B_lag = B[t-1] if t > 0 else ini.B

        chi[t] = ss.chi + par.omega*(B_lag-ss.B)/ss.Y
        B[t] = (1+rb[t])*B_lag - chi[t]


@nb.njit
def mutual_fund(par,ini,ss,r,ra,Div,pD,pi):

    for t_ in range(par.T):

        t = (par.T-1) - t_

        # p_eq
        p_eq_plus = next(pD,t,ss.pD)
        Div_plus = next(Div,t,ss.Div)
        pD[t] = (Div_plus+p_eq_plus) / (1+r[t])

    term_pD = pD[0]+Div[0]
    term_B = (1+ss.i)/(1+pi[0])*ini.B
    
    ra[0] = (term_B+term_pD)/ini.A_hh -1
    ra[1:] = r[:-1]


@nb.njit
def NKWC(par,ini,ss,pi_w,L,w,C_hh,NKWC_res):

    # a. phillips curve
    pi_w_plus = lead(pi_w,ss.pi_w)

    LHS = pi_w
    RHS = par.kappa*(par.varphi*L**par.nu - 1/par.mu*w*C_hh**(-par.sigma)) + par.beta*pi_w_plus
    NKWC_res[:] = LHS-RHS

@nb.njit
def market_clearing(par,ini,ss,Y,C_hh,A_hh,clearing_A,clearing_Y,pD,B):

    # a. market clearing
    clearing_A[:] = pD + B - A_hh
    clearing_Y[:] = Y - C_hh  
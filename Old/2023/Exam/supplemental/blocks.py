import numpy as np
import numba as nb

from GEModelTools import lag, lead, prev, next

@nb.njit
def production(par,ini,ss,w,TFP,px,delta,Vj,errors_Vj):

    # a. fixed
    TFP[:] = ss.TFP
    w[:] = ss.w
    delta[:] = ss.delta

    # b. Bellman
    Vj_plus = lead(Vj,ss.Vj)
    delta_plus = lead(delta,ss.delta)

    cont_Vj = (1-delta_plus)*par.beta_firm*Vj_plus
    errors_Vj[:] = Vj-(px*TFP-w+cont_Vj)

@nb.njit
def labor_market(par,ini,ss,v,S,delta,u,theta,lambda_v,lambda_u_s,errors_u):

    theta[:] = v/S

    lambda_v[:] = par.A*theta**(-par.alpha)
    lambda_u_s[:] = par.A*theta**(1-par.alpha)

    u_lag = lag(ini.u,u)        
    errors_u[:] = u - (u_lag-S*lambda_u_s+delta*(1-u_lag))

@nb.njit
def entry(par,ini,ss,Vj,lambda_v,errors_Vv):
    
    LHS = -par.kappa + lambda_v*Vj
    RHS = 0

    errors_Vv[:] = LHS-RHS

@nb.njit
def price_setters(par,ini,ss,px,pi,TFP,u,errors_pi):

    LHS = 1-par.epsilon + par.epsilon*px

    TFP_plus = lead(TFP,ss.TFP)
    pi_plus = lead(pi,ss.pi)        
    u_plus = lead(u,ss.u)
    
    Y = TFP*(1-u)
    Y_plus = TFP_plus*(1-u_plus)

    RHS = par.phi*pi*(1+pi) - par.phi*par.beta_firm*(pi_plus*(1+pi_plus)*Y_plus/Y)

    errors_pi[:] = LHS-RHS

@nb.njit
def central_bank(par,ini,ss,pi,i):

    i[:] = (1+ss.i)*((1+pi)/(1+ss.pi))**par.delta_pi - 1
    
@nb.njit
def dividends(par,ini,ss,TFP,u,w,div):

    div[:] = TFP*(1-u) - w*(1-u)

@nb.njit
def financial_market(par,ini,ss,pi,i,q,r):

    pi_plus = lead(pi,ss.pi)
    R_plus = (1+i)/(1+pi_plus)

    # a. price of government debt
    for k in range(par.T):
        
        t = par.T-1-k
        q_plus = next(q,t,ss.q)
        q[t] = (1+par.delta_q*q_plus)/R_plus[t]

    # b. real interest rate (ex post)
    r[0] = (1+par.delta_q*q[0])*ini.B/ini.A_hh - 1
    r[1:] = R_plus[:-1] - 1

@nb.njit
def government(par,ini,ss,G,U_UI_hh_guess,w,u,q,Phi,transfer,X,taut,tau,taxes,B):

    # a. expenses
    Phi[:] = par.phi_obar*w*U_UI_hh_guess + par.phi_ubar*w*(u-U_UI_hh_guess)
    transfer[:] = ss.transfer
    X[:] = Phi + G + transfer
    
    # b. taxes and debt
    pre_tax_hh_income = w*(1-u) + Phi

    for t in range(par.T):

        B_lag = prev(B,t,ini.B)

        taut[t] = ( (1+par.delta_q*q[t])*B_lag + X[t] - ss.q*ss.B ) / pre_tax_hh_income[t]
        tau[t] = par.omega*taut[t]+(1-par.omega)*ss.tau

        taxes[t] = tau[t]*pre_tax_hh_income[t]
        B[t] = ( (1+par.delta_q*q[t])*B_lag+X[t]-taxes[t])/q[t]
    
@nb.njit
def market_clearing(par,ini,ss,G,TFP,pi,i,C_hh,u,q,B,U_ALL_hh,U_UI_hh_guess,U_UI_hh,
                    Y,clearing_Y,qB,A_hh,errors_assets,errors_U,errors_U_UI):

    Y[:] = TFP*(1-u)

    # a. asset market clearing
    qB[:] = q*B
    errors_assets[:] = qB-A_hh

    # b. goods market clearing
    clearing_Y[:] = Y - (C_hh + G)

    # c. final targets
    errors_U[:] = u-U_ALL_hh
    errors_U_UI[:] = U_UI_hh_guess-U_UI_hh

@nb.njit
def ann(par,ini,ss,i,r,pi,i_ann,r_ann,pi_ann):

    for t in range(par.T):

        i_ann[t] = 1.0
        r_ann[t] = 1.0
        pi_ann[t] = 1.0

        for k in range(12):

            lag_i = prev(i,t-k,ini.i)
            lag_r = prev(r,t-k,ini.r)
            lag_pi = prev(pi,t-k,ini.pi)

            i_ann[t] *= (1+lag_i)
            r_ann[t] *= (1+lag_r)
            pi_ann[t] *= (1+lag_pi)

        i_ann[t] -= 1.0
        r_ann[t] -= 1.0
        pi_ann[t] -= 1.0            
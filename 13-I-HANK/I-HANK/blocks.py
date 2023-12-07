import numpy as np
import numba as nb
from GEModelTools import prev, next, lag, lead, isclose
from GEModelTools import lag, lead

##############
## Helpers ##
##############

@nb.njit
def price_index(P1,P2,eta,alpha):
    return (alpha*P1**(1-eta) + (1-alpha)*P2**(1-eta))**(1/(1-eta))

@nb.njit
def inflation_from_price(P,inival):

    P_lag = lag(inival,P) 
    pi = P/P_lag - 1

    return pi

@nb.njit
def price_from_inflation(P,pi,T,iniP):

    for t in range(T):
        if t == 0:
            P[t] = iniP*(1+pi[t]) 
        else:
            P[t] = P[t-1]*(1+pi[t]) 
   
############
## Blocks ##
############

@nb.njit
def production(par,ini,ss,
               ZT,ZNT,NT,NNT,piWT,piWNT,
               YT,YNT,WT,WNT,PTH,PNT):
    
    # a. production
    YT[:] = ZT*NT
    YNT[:] = ZNT*NNT
    
    # b. wages
    price_from_inflation(WT,piWT,par.T,ss.WT)
    price_from_inflation(WNT,piWNT,par.T,ss.WNT)

    # c. price = marginal cost
    PTH[:] = WT/ZT
    PNT[:] = WNT/ZNT

@nb.njit
def prices(par,ini,ss,
           PF_s,E,PTH,PNT,WT,WNT,
           PF,PTH_s,PT,P,Q,wT,wNT):
    
    # a. convert curency
    PF[:] = PF_s*E
    PTH_s[:] = PTH/E

    # b. price indices
    PT[:] = price_index(PF,PTH,par.etaF,par.alphaF)
    P[:] = price_index(PT,PNT,par.etaT,par.alphaT)

    # c. real exchange rate
    Q[:] = PF/P
 
    # d. real wage
    wT[:] = WT/P
    wNT[:] = WNT/P

@nb.njit
def inflation(par,ini,ss,
              PF_s,PF,PNT,PTH,PT,P,PTH_s,
              pi_F_s,pi_F,pi_NT,pi_TH,pi_T,pi,pi_TH_s):

    pi_F_s[:] = inflation_from_price(PF_s,ini.PF_s)
    pi_F[:] = inflation_from_price(PF,ini.PF)
    pi_NT[:] = inflation_from_price(PNT,ini.PNT)
    pi_TH[:] = inflation_from_price(PTH,ini.PTH)
    pi_T[:] = inflation_from_price(PT,ini.PT)
    pi[:] = inflation_from_price(P,ini.P)
    pi_TH_s[:] = inflation_from_price(PTH_s,ini.PTH_s)

@nb.njit
def central_bank(par,ini,ss,pi,i,r,ra):

    # a. taylor rule
    pi_plus = lead(pi,ss.pi)
    i[:] = ss.i + par.phi*pi_plus

    # b. fisher
    pi_plus = lead(pi,ss.pi)
    r[:] = (1+i)/(1+pi_plus)-1

    lag_i = lag(ini.i,i)
    ra[:] = (1+lag_i)/(1+pi)-1

@nb.njit
def government(par,ini,ss,
               PNT,P,wT,NT,wNT,NNT,ra,G,B,tau,inc_T,inc_NT):

    # a. government budget
    for t in range(par.T):

        tax_base = wT[t]*NT[t]+wNT[t]*NNT[t]
        
        B_lag = prev(B,t,ini.B)

        G[t] = ss.G
        tau[t] = ss.tau + par.omega*(B_lag-ss.B)/(ss.YT+ss.YNT)

        tax_base = wT[t]*NT[t]+wNT[t]*NNT[t]
        B[t] = (1+ra[t])*B_lag + PNT[t]/P[t]*G[t]-tau[t]*tax_base

    # b. household income
    inc_T[:] = (1-tau)*wT*NT/par.sT
    inc_NT[:] = (1-tau)*wNT*NNT/(1-par.sT)

@nb.njit
def NKWCs(par,ini,ss,piWT,piWNT,NT,NNT,wT,wNT,tau,UC_T_hh,UC_NT_hh,NKWCT_res,NKWCNT_res):

    # a. phillips curve tradeable
    piWT_plus = lead(piWT,ss.piWT)

    LHS = piWT
    RHS = par.kappa*(par.varphiT*(NT/par.sT)**par.nu-1/par.muw*(1-tau)*wT*UC_T_hh/par.sT) + par.beta*piWT_plus    
    
    NKWCT_res[:] = LHS-RHS

    # b. phillips curve non-tradeable
    piWNT_plus = lead(piWNT,ss.piWNT)

    LHS = piWNT
    RHS = par.kappa*(par.varphiNT*(NNT/(1-par.sT))**par.nu-1/par.muw*(1-tau)*wNT*UC_NT_hh/(1-par.sT)) + par.beta*piWNT_plus
    
    NKWCNT_res[:] = LHS-RHS

@nb.njit
def UIP(par,ini,ss,Q,r,rF,UIP_res):

    Q_plus = lead(Q,ss.Q)

    LHS = 1+r
    RHS = (1+rF)*Q_plus/Q
    UIP_res[:] = LHS-RHS

@nb.njit
def consumption(par,ini,ss,
                C_hh,PT,PNT,P,PTH,PF,M_s,PTH_s,PF_s,
                CT,CNT,CTF,CTH,CTH_s):

    # a. home - tradeable vs. non-tradeable
    CT[:] = par.alphaT*(PT/P)**(-par.etaT)*C_hh
    CNT[:] = (1-par.alphaT)*(PNT/P)**(-par.etaT)*C_hh

    # b. home - home vs. foreign tradeable
    CTF[:] = par.alphaF*(PF/PT)**(-par.etaF)*CT
    CTH[:] = (1-par.alphaF)*(PTH/PT)**(-par.etaF)*CT

    # c. foreign - home tradeable
    CTH_s[:] = (PTH_s/PF_s)**(-par.eta_s)*M_s

@nb.njit
def market_clearing(par,ini,ss,
             YT,CTH,CTH_s,YNT,CNT,G,
             clearing_YT,clearing_YNT):
    
    clearing_YT[:] = YT-CTH-CTH_s
    clearing_YNT[:] = YNT-CNT-G

@nb.njit
def accounting(par,ini,ss,
               PT,YT,PNT,YNT,P,C_hh,G,A_hh,B,ra,
               GDP,NX,CA,NFA,Walras):
    
    GDP[:] = (PT*YT+PNT*YNT)/P 
    NX[:] = GDP-C_hh-PNT/P*G

    NFA[:] = A_hh-B

    NFA_lag = lag(ini.NFA,NFA)
    CA[:] = NX + ra*NFA_lag

    Walras[:] = (NFA-NFA_lag) - CA
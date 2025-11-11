import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def production(par,ini,ss,L,Y):
    Y[:] = L


@nb.njit
def central_bank(par,ini,ss,r):
    r[:] = ss.r 


@nb.njit
def government(par,ini,ss,Taxes,B,Y,r,transfers,tax):

    
    for t in range(par.T):
        B_lag = B[t-1] if t > 0 else ini.B
        tax[t] = ss.tax + par.phi_debt*(B_lag-ss.B)

        Taxes[t] = tax[t] * Y[t]
        B[t] = (1+r[t]) * B_lag + par.G + transfers[t] - Taxes[t] 
                
@nb.njit
def RA_HHs(par,ini,ss, L, C_hh, A_hh, MUC_hh, r, transfers, tax, C_R):

    if par.HH_type == 'RANK':

        # solve euler equation 
        for k in range(par.T):

            t = par.T - k - 1
            if k==0: # last period
                r_p = ss.r
                CR_p = ss.C_R 
            else:
                r_p = r[t+1]
                CR_p = C_R[t+1]

            C_R[t] = # fill up here
        
        # aggregate
        C_hh[:] = C_R

        # solve for savings using budget constraint
        for t in range(par.T):
            if t==0:
                A_lag = ini.A_hh
            else:
                A_lag = A_hh[t-1] 
            
            A_hh[t] = # fill up here

        # MUC_hh[:] = C_hh**(-par.sigma)
        MUC_hh[:] = C_R**(-par.sigma)

@nb.njit
def TA_HHs(par,ini,ss,L, C_hh, A_hh, MUC_hh, r, C_R, C_HtM, transfers, tax):

    if par.HH_type == 'TANK':

        # solve euler equation 
        for k in range(par.T):

            t = par.T - k - 1
            if k==0:
                r_p = ss.r
                CR_p = ss.C_R 
            else:
                r_p = r[t+1]
                CR_p = C_R[t+1]

            C_R[t] = # fill up here
        
        # Solve HtM consumption 
        C_HtM[:] = (1-tax)*L + transfers

        # aggregate
        C_hh[:] = C_R*(1-par.sHtM) + C_HtM*par.sHtM

        # solve for savings using budget constraint
        for t in range(par.T):
            if t==0:
                A_lag = ini.A_hh
            else:
                A_lag = A_hh[t-1] 
            
            A_hh[t] = # fill up here


        # MUC_hh[:] = C_hh**(-par.sigma)
        MUC_hh[:] = # fill up here



@nb.njit
def labor_supply(par,ini,ss,L,MUC_hh,pi,NKPC_res,tax):

    # a. labor supply 
    for t in range(par.T):

        LHS = pi[t]
        wedge = par.varphi*L[t]**par.nu - 1/par.mu * (1-tax[t])*MUC_hh[t]
        if t < par.T-1:
            RHS = par.kappa * wedge + par.beta * pi[t+1]
        else:
            RHS = par.beta * wedge
        NKPC_res[t] = LHS-RHS


@nb.njit
def market_clearing(par,ini,ss,B,Y,C_hh,A_hh,A,clearing_A,clearing_Y):
        
    # a. aggregates
    A[:] = B

    # b. market clearing
    clearing_A[:] = A-A_hh
    clearing_Y[:] = Y-C_hh-par.G
    
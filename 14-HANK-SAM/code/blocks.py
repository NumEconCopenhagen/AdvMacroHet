import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def delta_func(Vj,par,ss):    
    """ separations """

    if par.exo_sep:
        return ss.delta*np.ones(Vj.size)
    else:
        return par.p*(np.fmax(Vj/par.Upsilon,1))**(-par.psi)

@nb.njit
def mu_func(Vj,par):
    """ continuation costs """

    if par.exo_sep:
        
        return np.zeros(Vj.size)
    
    else:

        if np.abs(par.psi-1.0) < 1e-8:
            _fac = np.log(np.fmax(Vj/par.Upsilon,1.0))
        else:
            _fac = par.psi/(par.psi-1)*(1.0-(np.fmax(Vj/par.Upsilon,1.0))**(1-par.psi))

        _nom = par.p*par.Upsilon
        _denom = 1.0-par.p*(np.fmax(Vj/par.Upsilon,1.0))**(-par.psi)

        return _fac*_nom/_denom   

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):
    
    block_pre_SAM(par,ini,ss,path,ncols=ncols)

    if not par.only_SAM:
        block_pre_HANK(par,ini,ss,path,ncols=ncols)

@nb.njit
def block_pre_SAM(par,ini,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    for ncol in range(ncols):

        # unpack
        delta = path.delta[ncol,:]
        entry = path.entry[ncol,:]
        lambda_u = path.lambda_u[ncol,:]
        lambda_v = path.lambda_v[ncol,:]
        M = path.M[ncol,:]
        mu = path.mu[ncol,:]
        px = path.px[ncol,:]
        shock_TFP = path.shock_TFP[ncol,:]
        theta = path.theta[ncol,:]
        u = path.u[ncol,:]
        ut = path.ut[ncol,:]
        v = path.v[ncol,:]
        Vj = path.Vj[ncol,:]
        vt = path.vt[ncol,:]
        Vv = path.Vv[ncol,:]
        w = path.w[ncol,:]

        wage_subsidy = path.wage_subsidy[ncol,:]
        hiring_subsidy = path.hiring_subsidy[ncol,:]

        errors_ut = path.errors_ut[ncol,:]
        errors_Vj = path.errors_Vj[ncol,:]
        errors_vt = path.errors_vt[ncol,:]
        errors_Vv = path.errors_Vv[ncol,:]
        errors_WageRule = path.errors_WageRule[ncol,:]

        # i. wage and profits
        if par.wage_setting == 'fixed':
            w[:] = ss.w

        M[:] = shock_TFP*px-(w-wage_subsidy)

        # ii. Vj
        delta[:] = delta_func(Vj,par,ss)
        mu[:] = mu_func(Vj,par)

        Vj_plus = lead(Vj,ss.Vj)
        delta_plus = lead(delta,ss.delta)
        mu_plus = lead(mu,ss.mu)

        cont_Vj = (1-delta_plus)*par.beta_firm*Vj_plus - par.beta_firm*mu_plus
        errors_Vj[:] = Vj-(M+cont_Vj)

        if not par.free_entry:   
            entry[:] = ss.entry*(np.fmax(Vv[:]/ss.Vv,0.0))**par.xi

        # iii. labor market variables
        theta[:] = vt/ut

        lambda_v[:] = par.A*theta**(-par.alpha)
        lambda_u[:] = par.A*theta**(1-par.alpha)

        u[:] = (1-lambda_u)*ut
        v[:] = (1-lambda_v)*vt

        u_lag = lag(ini.u,u)
        v_lag = lag(ini.v,v)

        errors_vt[:] = vt - ((1-ss.delta)*v_lag + entry)
        errors_ut[:] = ut - (u_lag + delta*(1-u_lag))

        # iv. Vv
        if par.free_entry:

            LHS = np.zeros(par.T)
            Vv_plus = np.zeros(par.T)

        else:

            LHS = Vv
            Vv_plus = lead(Vv,ss.Vv)
        
        Vj_new_hire = Vj + hiring_subsidy
        RHS = -par.kappa + lambda_v*Vj_new_hire + (1-lambda_v)*(1-ss.delta)*par.beta_firm*Vv_plus

        errors_Vv[:] = LHS-RHS

        # v. wage rule
        if par.wage_setting == 'fixed':

            pass

        else:

            for t in range(par.T):
                
                w_lag = w[t-1] if t > 0 else ini.w

                curr_w = (u[t]/ss.u)**par.eta_u*( (1.0-u[t])/(1.0-ss.u))**par.eta_e*(shock_TFP[t])**par.eta_TFP
                RHS = ss.w*(w_lag/ss.w)**par.rho_w*curr_w**(1-par.rho_w)

                errors_WageRule[t] = w[t]-RHS

@nb.njit
def block_pre_HANK(par,ini,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    for ncol in range(ncols):

        # unpack
        B = path.B[ncol,:]
        G = path.G[ncol,:]
        hiring_subsidy = path.hiring_subsidy[ncol,:]
        lambda_v = path.lambda_v[ncol,:]
        phi_obar = path.phi_obar[ncol,:]
        Pi = path.Pi[ncol,:]
        public_transfer = path.public_transfer[ncol,:]
        px = path.px[ncol,:]
        q = path.q[ncol,:]
        qB = path.qB[ncol,:]
        R = path.R[ncol,:]
        RealR = path.RealR[ncol,:]
        RealR_ex_post = path.RealR_ex_post[ncol,:]
        shock_TFP = path.shock_TFP[ncol,:]
        tau = path.tau[ncol,:]
        transfer = path.transfer[ncol,:]
        u = path.u[ncol,:]
        U_UI_hh_guess = path.U_UI_hh_guess[ncol,:]
        UI = path.UI[ncol,:]
        vt = path.vt[ncol,:]
        w = path.w[ncol,:]
        wage_subsidy = path.wage_subsidy[ncol,:]
        Yt_hh = path.Yt_hh[ncol,:]

        errors_Pi = path.errors_Pi[ncol,:]

        # i. dividence
        transfer[:] = par.div_hh*(shock_TFP-w)*(1-u) + public_transfer

        # ii. Phillips curve
        LHS = 1-par.epsilon_p + par.epsilon_p*px

        Pi_plus = lead(Pi,ss.Pi)        
        shock_TFP_plus = lead(shock_TFP,ss.shock_TFP)
        u_plus = lead(u,ss.u)

        RHS = par.phi*(Pi-ss.Pi)*Pi - par.beta_firm*par.phi*((Pi_plus-ss.Pi)*Pi_plus*(shock_TFP_plus*u_plus)/(shock_TFP*u))

        errors_Pi[:] = LHS-RHS

        # iii. Taylor rule and Fisher rule   
        for t in range(par.T):
            R_lag = ss.R if t == 0 else R[t-1]
            R[t] = ss.R*(R_lag/ss.R)**(par.rho_R)*(Pi[t]/ss.Pi)**(par.delta_pi*(1-par.rho_R))
            
            if t < par.T-1:
                RealR[t] = R[t]/Pi[t+1]
            else:
                RealR[t] = R[t]/ss.Pi

        # iv. arbitrage
        for k in range(par.T):
            t = par.T-1-k
            q_plus = q[t+1] if t < par.T-1 else ss.q
            q[t] = (1+par.delta_q*q_plus)/RealR[t]

        q_lag = lag(ini.q,q)
        RealR_ex_post[:] = (1+par.delta_q*q)/q_lag

        # v. fiscal policy
        UI[:] = phi_obar*w*U_UI_hh_guess + par.phi_ubar*w*(u-U_UI_hh_guess)
        Yt_hh[:] = w*(1-u) + UI

        for t in range(par.T):
            
            B_lag = B[t-1] if t > 0 else ini.B
            tau[t] = ss.tau + par.omega*ss.q*(B_lag-ss.B)/ss.Yt_hh

            expenses = UI[t] 
            expenses += G[t] 
            expenses += public_transfer[t] 
            expenses += wage_subsidy[t]*(1-u[t])
            expenses += hiring_subsidy[t]*lambda_v[t]*vt[t]

            B[t] = ((1+par.delta_q*q[t])*B_lag + expenses - tau[t]*Yt_hh[t])/q[t]

        qB[:] = q*B
        
@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    block_post_SAM(par,ini,ss,path,ncols=ncols)
        
    if not par.only_SAM:
        block_post_HANK(par,ini,ss,path,ncols=ncols)

@nb.njit
def block_post_SAM(par,ini,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    pass

@nb.njit
def block_post_HANK(par,ini,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    for ncol in range(ncols):

        # unpack
        A_hh = path.A_hh[ncol,:]
        qB = path.qB[ncol,:]
        U_UI_hh = path.U_UI_hh[ncol,:]
        U_UI_hh_guess = path.U_UI_hh_guess[ncol,:]

        errors_assets = path.errors_assets[ncol,:]
        errors_U_UI = path.errors_U_UI[ncol,:]

        # i. asset market clearing
        errors_assets[:] = qB-A_hh

        # ii. share on unemployment benefits
        errors_U_UI[:] = U_UI_hh_guess-U_UI_hh
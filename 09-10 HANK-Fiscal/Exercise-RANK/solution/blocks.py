import numba as nb

from GEModelTools import lead

@nb.njit
def central_bank(par,ini,ss,r):

    # real rate rule: 1+i_t = (1+r_ss)(1+pi_t+1)
    r[:] = ss.r


@nb.njit
def government(par,ini,ss,G,Y,r,B,taxes,tau,Z):

    for t in range(par.T):

        B_lag = B[t-1] if t > 0 else ini.B
        r_lag = r[t-1] if t > 0 else ini.r

        # a. debt rule
        B[t] = ss.B + par.rho_B*(B_lag-ss.B+G[t]-ss.G)

        # b. taxes from the budget
        taxes[t] = (1+r_lag)*B_lag + G[t] - B[t]

    # c. tax rate and post-tax income (mu = 1, so w = 1 and L = Y)
    tau[:] = taxes/Y
    Z[:] = (1-tau)*Y


@nb.njit
def households(par,ini,ss,Z,r,C_R,C_H,C,A_R,Euler):

    # a. hand-to-mouth
    C_H[:] = Z

    # b. Ricardian
    C_R_plus = lead(C_R,ss.C_R)
    Euler[:] = C_R**(-par.sigma) - par.beta*(1+r)*C_R_plus**(-par.sigma)

    for t in range(par.T):
        A_R_lag = A_R[t-1] if t > 0 else ini.A_R
        r_lag = r[t-1] if t > 0 else ini.r
        A_R[t] = (1+r_lag)*A_R_lag + Z[t] - C_R[t]

    # c. aggregate
    C[:] = par.lam*C_H + (1-par.lam)*C_R


@nb.njit
def market_clearing(par,ini,ss,Y,C,G,A_R,B,clearing_Y,clearing_A):

    clearing_Y[:] = Y-C-G
    clearing_A[:] = (1-par.lam)*A_R-B


@nb.njit
def unions(par,ini,ss,Y,C,tau,pi_w,pi):

    # a. W-NKPC solved backwards from steady state (L = Y and w = 1)
    for k in range(par.T):
        t = par.T-1-k
        pi_w_plus = pi_w[t+1] if t < par.T-1 else ss.pi_w
        pi_w[t] = par.kappa*(par.varphi*Y[t]**par.nu - (1-tau[t])*C[t]**(-par.sigma)) + par.beta*pi_w_plus

    # b. constant real wage, so price inflation equals wage inflation
    pi[:] = pi_w


@nb.njit
def fisher(par,ini,ss,r,pi,i):

    pi_plus = lead(pi,ss.pi)
    i[:] = (1+r)*(1+pi_plus)-1

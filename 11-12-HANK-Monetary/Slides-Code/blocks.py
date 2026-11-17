import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def production(par,ini,ss,L,Y):

    Y[:] = L


@nb.njit
def central_bank(par,ini,ss,eps_r,r):

    # real rate rule
    r[:] = ss.r + eps_r
        

@nb.njit
def government(par,ini,ss,r,Y,L,B,ra,taxes,tau,Z):

    # a. real bonds: the return paid at t is the rate set at t-1
    ra[:] = lag(ss.r,r)

    # b. debt is fixed, taxes pay the interest bill
    B[:] = ss.B
    taxes[:] = ra*ss.B + par.G
    tau[:] = taxes/Y

    # c. household income after tax (wage = 1, no markups)
    Z[:] = (1-tau)*L


@nb.njit
def NKWC(par,ini,ss,pi_w,L,tau,C_hh,NKWC_res):

    pi_w_plus = lead(pi_w,ss.pi_w)

    LHS = pi_w
    RHS = par.kappa*(par.varphi*L**par.nu - (1-tau)*C_hh**(-par.sigma)) + par.beta*pi_w_plus
    NKWC_res[:] = LHS-RHS


@nb.njit
def market_clearing(par,ini,ss,B,A_hh,Y,C_hh,clearing_A,clearing_Y):
        
    clearing_A[:] = B-A_hh
    clearing_Y[:] = Y-C_hh-par.G

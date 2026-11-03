def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. central bank and government
    ss.r = par.r_target_ss
    ss.Y = 1.0
    ss.G = par.G_target_ss
    ss.B = par.B_target_ss
    ss.taxes = ss.r*ss.B + ss.G
    ss.tau = ss.taxes/ss.Y
    ss.Z = (1-ss.tau)*ss.Y

    # b. households: Ricardian hold all the debt
    par.beta = 1/(1+ss.r)
    ss.C_H = ss.Z
    ss.A_R = ss.B/(1-par.lam)
    ss.C_R = ss.Z + ss.r*ss.A_R
    ss.C = par.lam*ss.C_H + (1-par.lam)*ss.C_R
    ss.Euler = 0.0

    # c. market clearing
    ss.clearing_Y = ss.Y-ss.C-ss.G
    ss.clearing_A = (1-par.lam)*ss.A_R-ss.B

    # d. unions: zero inflation pins down varphi
    ss.pi_w = ss.pi = 0.0
    ss.i = ss.r
    par.varphi = (1-ss.tau)*ss.C**(-par.sigma)/ss.Y**par.nu

    if do_print:
        print(f' beta = {par.beta:8.4f}')
        print(f' tau  = {ss.tau:8.4f}')
        print(f' C_R  = {ss.C_R:8.4f}, C_H = {ss.C_H:8.4f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')

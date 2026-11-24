import os
import sys

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import optimize
from consav.linear_interp import interp_1d_vec

model_folder = os.path.abspath('../09-10 HANK-Fiscal/Exercise-HANK/solution')
sys.path.insert(0, model_folder)

import HANKModel
import steady_state

"""
Figures for the monetary policy slides (run from this folder), bond HANK of lecture 10, annual model
In the code r[t] is the return paid at t, which is r_{t-1} in the slides, so slide date s is code index s+1
"""

fig_folder = 'tex/figs'
T_plot = 11
sigma_base, B_base = 2.0, 0.25
omega_rule = 0.35
zeta_procyclical = 0.3
rho_shock = 0.4
Bs = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
horizons = np.arange(1, 22)

plt.rcParams.update({'axes.grid': True, 'grid.color': 'black', 'grid.alpha': 0.25, 'grid.linestyle': '--', 'font.size': 12, 'lines.linewidth': 2})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

"""
Solvers
"""

@nb.njit
def solve_hh_backwards_zeta(par, z_trans, ra, Z, vbeg_a_plus, vbeg_a, a, c):
    """EGM step where the income share z**(1+zeta*log(Z/Z_ss)) moves with aggregate income."""

    z_share = par.z_grid**(1 + par.zeta*np.log(Z/par.Z_ss))
    z_share = z_share/np.sum(par.z_ergodic*z_share)

    for i_fix in range(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):

            c_endo = (par.beta_grid[i_z]*vbeg_a_plus[i_fix, i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid
            m = (1 + ra)*par.a_grid + Z*z_share[i_z]
            interp_1d_vec(m_endo, par.a_grid, m, a[i_fix, i_z])
            a[i_fix, i_z, :] = np.fmax(a[i_fix, i_z, :], 0.0)
            c[i_fix, i_z] = m - a[i_fix, i_z]

        # b. expectation step
        v_a = (1 + ra)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a


def solve_hank(sigma, B, zeta=0.0):
    """Steady state and household Jacobians for one calibration of the IKC model."""

    # a. steady state, Nz = 7 productivity states times 2 discount factor states
    model = HANKModel.HANKModelClass(name='baseline', par={'model_type': 'IKC', 'zeta': zeta, 'Z_ss': 1.0, 'z_ergodic': np.zeros(14)})
    model.par.sigma = sigma
    model.par.B_target_ss = model.par.A_target_ss = B
    root = optimize.root_scalar(steady_state.obj_ss, bracket=(0.75, 1/(1 + model.par.r_target_ss) - 1e-4), method='brentq', args=(model,)).root
    steady_state.obj_ss(root, model)

    # b. Jacobians, with income shares that move with Z
    model.par.Z_ss = model.ss.Z
    model.par.z_ergodic[:] = model.ss.Dbeg[0].sum(axis=1)
    model.solve_hh_backwards = solve_hh_backwards_zeta
    model._compute_jac_hh()

    return model, np.array(model.jac_hh[('C_hh', 'Z')]), np.array(model.jac_hh[('C_hh', 'ra')])


def ikc_fixed_B(M, M_r, B, dr):
    """Solve the IKC with taxes adjusting every period, closed by dY=0 at T."""

    T = M.shape[0]
    lhs = np.eye(T) - M
    rhs = M_r@dr - B*M@dr
    lhs[-1, :] = 0.
    lhs[-1, -1] = 1.
    rhs[-1] = 0.

    return np.linalg.solve(lhs, rhs)


def ikc_tax_rule(M, M_r, ss, omega, dr):
    """Solve the IKC with the debt rule, proportional taxes and fixed G."""

    T = M.shape[0]
    Lag = np.eye(T, k=-1)
    debt_map = np.linalg.inv(np.eye(T) - (1 + ss.r - omega)*Lag)
    T_dY = ss.tau*(np.eye(T) - omega*Lag@debt_map)
    T_dr = omega*ss.B*Lag@debt_map

    return np.linalg.solve(np.eye(T) - M + M@T_dY, M_r@dr - M@T_dr@dr)


def rank_Mr(ss, sigma, T):
    """Closed form for the RANK Jacobian to the return paid at s, s>=1, with Z fixed."""

    beta = 1/(1 + ss.r)
    substitution = beta*ss.C_hh/sigma
    M_r = np.outer(np.ones(T), beta**np.arange(T)*((1 - beta)*ss.B - substitution))
    M_r += substitution*(np.arange(T)[:, None] >= np.arange(T)[None, :])
    M_r[:, 0] = (1 - beta)*ss.B

    return M_r


def rank_total(ss, sigma, dr):
    """RANK consumption response from the Euler equation, with dC=dY."""

    return -ss.C_hh/sigma/(1 + ss.r)*np.append(np.cumsum(dr[::-1])[::-1][1:], 0.)


"""
Baseline
"""

model, M, M_r = solve_hank(sigma_base, B_base)
par, ss = model.par, model.ss
T = par.T

dr = np.zeros(T)
dr[1:] = -0.0025*rho_shock**np.arange(T - 1)

q = (1 + ss.r)**(-np.arange(T))
beta_rank = 1/(1 + ss.r)
M_rank = (1 - beta_rank)*np.outer(np.ones(T), q)
M_r_rank = rank_Mr(ss, sigma_base, T)

dC_rank = rank_total(ss, sigma_base, dr)
direct_rank = M_r_rank@dr
indirect_rank = M_rank@(dC_rank - B_base*dr)

dC_hank = ikc_fixed_B(M, M_r, B_base, dr)
direct_hank = M_r@dr
indirect_hank = dC_hank - direct_hank

print(f'beta = {par.beta_H:.4f}, C = {ss.C_hh:.3f}, tau = {ss.tau:.3f}')
print(f'RANK check: {np.abs(dC_rank - direct_rank - indirect_rank)[:10].max():.1e}')
print(f'RANK direct share {direct_rank[0]/dC_rank[0]:.2f}, HANK direct share {direct_hank[0]/dC_hank[0]:.2f}')
print(f'iMPC00 RANK {M_rank[0, 0]:.3f}, HANK {M[0, 0]:.3f}')
print(f'dC0 RANK {100*dC_rank[0]:.3f}, HANK {100*dC_hank[0]:.3f}')

"""
Lecture 11: Jacobians
"""

quarters = np.arange(T_plot)
columns = [0, 1, 2, 3, 5]

fig, ax = plt.subplots(figsize=(7, 3.6))

for s in columns:
    ax.plot(quarters, M_r_rank[:T_plot, s + 1], label=f's = {s}')

ax.set(xlabel='Years', title='Columns of $M^{r}$ in RANK')
ax.legend(ncol=5, fontsize=9)
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_Mr_RANK.pdf')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)

for ax, Jacobian, title in zip(axes, [M_r_rank, M_r], ['RANK', 'HANK']):

    for s in columns:
        ax.plot(quarters, Jacobian[:T_plot, s + 1], label=f's = {s}')

    ax.set(xlabel='Years', title=title)

axes[0].legend(ncol=5, fontsize=9)
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_Mr_compare.pdf')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(quarters, M_rank[:T_plot, 0], label='RANK')
axes[0].plot(quarters, M[:T_plot, 0], label='HANK')
axes[0].set(xlabel='Years', title='Response to income at $s=0$: column of $M$')
axes[0].legend()
axes[1].plot(quarters, -0.01*M_r_rank[:T_plot, 1], label='RANK')
axes[1].plot(quarters, -0.01*M_r[:T_plot, 1], label='HANK')
axes[1].set(xlabel='Years', title='Response to a rate cut at $s=0$: column of $M^{r}$')
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_M_vs_Mr.pdf')

"""
Lecture 11: decomposition
"""

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(quarters, 100*dC_rank[:T_plot], color='black', label='Total')
ax.plot(quarters, 100*direct_rank[:T_plot], label='Direct')
ax.plot(quarters, 100*indirect_rank[:T_plot], ls='--', label='Indirect')
ax.set(xlabel='Years', ylabel='% of steady state output', title='RANK')
ax.legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_decomp_RANK.pdf')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)

for ax, total, direct, indirect, title in zip(axes, [dC_rank, dC_hank], [direct_rank, direct_hank], [indirect_rank, indirect_hank], ['RANK', 'HANK']):
    ax.plot(quarters, 100*total[:T_plot], color='black', label='Total')
    ax.plot(quarters, 100*direct[:T_plot], label='Direct')
    ax.plot(quarters, 100*indirect[:T_plot], ls='--', label='Indirect')
    ax.set(xlabel='Years', title=title)

axes[0].set_ylabel('% of steady state output')
axes[0].legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_decomp_compare.pdf')

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(quarters, 100*dC_rank[:T_plot], label='RANK')
ax.plot(quarters, 100*dC_hank[:T_plot], label='HANK')
ax.set(xlabel='Years', ylabel='% of steady state output', title='Consumption response to a rate cut')
ax.legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_total_compare.pdf')

"""
Lecture 11: how robust is equivalence
"""

impact_fixed, cumulative_fixed, share_fixed, impact_rule, cumulative_rule = {}, {}, {}, {}, {}

for sigma in [1.0, 2.0]:

    impact_fixed[sigma], cumulative_fixed[sigma], share_fixed[sigma], impact_rule[sigma], cumulative_rule[sigma] = [], [], [], [], []

    for B in Bs:

        model_B, M_B, M_r_B = solve_hank(sigma, B)
        dC_R = rank_total(model_B.ss, sigma, dr)
        dC_fixed = ikc_fixed_B(M_B, M_r_B, B, dr)
        dC_rule = ikc_tax_rule(M_B, M_r_B, model_B.ss, omega_rule, dr)

        impact_fixed[sigma].append(dC_fixed[0]/dC_R[0])
        cumulative_fixed[sigma].append(dC_fixed[:10].sum()/dC_R[:10].sum())
        share_fixed[sigma].append((M_r_B@dr)[0]/dC_fixed[0])
        impact_rule[sigma].append(dC_rule[0]/dC_R[0])
        cumulative_rule[sigma].append(dC_rule[:10].sum()/dC_R[:10].sum())

    print(f'sigma = {sigma}')
    print('  impact, B fixed  ', np.round(impact_fixed[sigma], 2))
    print('  sum, B fixed     ', np.round(cumulative_fixed[sigma], 2))
    print('  direct share     ', np.round(share_fixed[sigma], 2))
    print('  impact, rule     ', np.round(impact_rule[sigma], 2))
    print('  sum, rule        ', np.round(cumulative_rule[sigma], 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)

for ax, series, title in zip(axes, [impact_fixed, cumulative_fixed], ['Impact', 'Sum over 10 years']):

    for sigma in [1.0, 2.0]:
        ax.plot(Bs, series[sigma], marker='o', label=f'$\\sigma={sigma:.0f}$')

    ax.axhline(1, color='black', lw=1)
    ax.set(xscale='log', xlabel='$B$, annual output', title=title, ylim=(0, 2))

axes[0].set_ylabel('HANK / RANK')
axes[0].legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_ratio_B.pdf')

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(Bs, impact_fixed[sigma_base], marker='o', label='$B$ held fixed')
ax.plot(Bs, impact_rule[sigma_base], marker='s', ls='--', label=f'Tax rule, $\\omega={omega_rule}$')
ax.axhline(1, color='black', lw=1)
ax.set(xscale='log', xlabel='$B$, annual output', ylabel='HANK / RANK, impact', ylim=(0, 1.3))
ax.legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/mon_ratio_omega.pdf')

"""
Lecture 12: forward guidance
"""

# a. RANK: nominal rate cut with inflation responding, solved backwards
kappa_slope = par.kappa*ss.C_hh**(-sigma_base)/par.mu*(par.nu + sigma_base/ss.C_hh)
impact_nominal = []

for s in horizons:

    dC_next, dpi_next = 0., 0.

    for t in reversed(range(s + 1)):
        di = -0.0025 if t == s - 1 else 0.
        dC_now = dC_next - ss.C_hh/sigma_base*beta_rank*(di - dpi_next)
        dpi_now = kappa_slope*dC_now + beta_rank*dpi_next
        dC_next, dpi_next = dC_now, dpi_now

    impact_nominal.append(dC_now)

impact_nominal = np.array(impact_nominal)
impact_real = ss.C_hh/sigma_base*beta_rank*0.0025*np.ones(len(horizons))
print('nominal/real at horizon 0, 2, 5:', np.round((impact_nominal/impact_real)[[0, 2, 5]], 2))

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(horizons[:11] - 1, 100*impact_real[:11], label='Real rate path given')
ax.plot(horizons[:11] - 1, 100*impact_nominal[:11], label='Nominal rate path given')
ax.set(yscale='log', xlabel='Horizon of the one-year rate cut, years', ylabel='Impact on $C$, % of $Y_{ss}$')
ax.legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/fg_RANK.pdf')

# b. HANK: real rate cut at each horizon, B fixed, acyclical and procyclical risk
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(horizons - 1, np.ones(len(horizons)), color='black', label='RANK')

for zeta, label in [(0.0, 'HANK, acyclical risk'), (zeta_procyclical, 'HANK, procyclical risk')]:

    model_zeta, M_zeta, M_r_zeta = solve_hank(sigma_base, B_base, zeta)
    impact = []

    for s in horizons:
        dr_s = np.zeros(T)
        dr_s[s] = -0.0025
        impact.append(ikc_fixed_B(M_zeta, M_r_zeta, B_base, dr_s)[0])

    impact_R = rank_total(model_zeta.ss, sigma_base, dr_s)[0]
    ax.plot(horizons - 1, np.array(impact)/impact_R, label=label)
    print(f'zeta={zeta}: HANK/RANK at 0,2,5,10,20:', np.round((np.array(impact)/impact_R)[[0, 2, 5, 10, 20]], 2))

ax.set(xlabel='Horizon of the one-year rate cut, years', ylabel='HANK / RANK, impact on $C$', ylim=(0, 1.2))
ax.legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/fg_HANK.pdf')

# c. partial equilibrium response to far-away rate cuts
fig, ax = plt.subplots(figsize=(7, 3.6))

for s, color in zip([5, 10], colors):
    ax.plot(np.arange(21), -0.01*M_r_rank[:21, s + 1], color=color, ls='--', label=f'RANK, $s={s}$')
    ax.plot(np.arange(21), -0.01*M_r[:21, s + 1], color=color, label=f'HANK, $s={s}$')

ax.set(xlabel='Years', title='Direct response to a rate cut at date $s$')
ax.legend(ncol=2, fontsize=9, loc='upper right')
fig.tight_layout()
fig.savefig(f'{fig_folder}/fg_Mr_far.pdf')

"""
Lecture 12: investment Jacobians (slide timing, r_t from t to t+1)
"""

# a. ARS (2024) calibration, annual, with mu = 1 and Y = 1
r_inv, delta, epsilon_I, K_Y = 0.05, 0.08, 4.0, 2.26
alpha_mu = (r_inv + delta)*K_Y
T_inv = 300

# b. linearized capital condition stacked over t = 0..T-1, with dK_{-1} = dK_T = 0
a_inv = delta*epsilon_I*(r_inv + delta)
A = (2 + r_inv + a_inv)*np.eye(T_inv) - (1 + r_inv)*np.eye(T_inv, k=-1) - np.eye(T_inv, k=1)
net_investment = np.eye(T_inv) - (1 - delta)*np.eye(T_inv, k=-1)
I_Y = delta*epsilon_I*alpha_mu*net_investment@np.linalg.solve(A, np.eye(T_inv, k=1))
I_r = -delta*epsilon_I*K_Y*net_investment@np.linalg.inv(A)

# c. columns
years = np.arange(16)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

for s in [1, 5, 10]:
    axes[0].plot(years, I_Y[:len(years), s], label=f's = {s}')
    axes[1].plot(years, -I_r[:len(years), s], label=f's = {s}')

axes[0].set(xlabel='Years', ylabel='$\\partial I_t/\\partial Y_s$', title='Columns of $I^{Y}$: news that $Y_s$ rises')
axes[1].set(xlabel='Years', ylabel='% of steady state output', title='Columns of $-I^{r}$: news of a 1pp cut in $r_s$')
axes[0].legend()
fig.tight_layout()
fig.savefig(f'{fig_folder}/inv_jacobians.pdf')

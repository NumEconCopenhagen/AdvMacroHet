import numpy as np
import matplotlib.pyplot as plt

import HANKModel

"""
Figures for the IKC, TANK and HANK slides (run from this folder)
"""

folder = '../../tex/figs'
columns = [0, 2, 4, 6, 8]
T_max = 12
quarters = np.arange(T_max)

plt.rcParams.update({'axes.grid': True, 'grid.color': 'black', 'grid.alpha': 0.25, 'grid.linestyle': '--'})

model = HANKModel.HANKModelClass(name='IKC',par={'model_type':'IKC'})
model.find_ss()
model.compute_jacs()
model.find_IRFs(shocks=['G'], do_print=False)

par, ss = model.par, model.ss
q = (1 + ss.r)**(-np.arange(par.T))

M_HANK = np.array(model.jac_hh[('C_hh', 'Z')])
beta_RANK = 1/(1 + ss.r)
M_RANK = (1 - beta_RANK)*np.outer(np.ones(par.T), q)

# TANK with the same quarterly MPC as HANK
lambda_TANK = (M_HANK[0, 0] - (1 - beta_RANK))/beta_RANK
M_TANK = (1 - lambda_TANK)*M_RANK + lambda_TANK*np.eye(par.T)

G, T, C, Y, B = [model.IRF[varname][:par.T] for varname in ['G', 'taxes', 'C_hh', 'Y', 'B']]

"""
Columns of M
"""

def plot_columns(ax, M, title):
    """Plot selected columns of an iMPC matrix."""

    for s in columns:
        ax.plot(quarters, M[:T_max, s], lw=2, label=f's = {s}')

    ax.set(title=title, xlabel='Years', ylim=(-0.01, 0.4))


for M, filename, title in [(M_RANK, 'iMPC_RANK', 'RANK'), (M_TANK, 'iMPC_TANK', 'TANK')]:

    fig, ax = plt.subplots(figsize=(7, 3.6))
    plot_columns(ax, M, title)
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{folder}/{filename}.pdf')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

for ax, M, title in zip(axes, [M_RANK, M_TANK, M_HANK], ['RANK', 'TANK', 'HANK']):
    plot_columns(ax, M, title)

axes[2].legend(fontsize=9)
fig.tight_layout()
fig.savefig(f'{folder}/iMPC_compare.pdf')

"""
First column of M at the annual frequency
"""

years = np.arange(6)
M_annual = M_HANK[:6, 0]

fig, ax = plt.subplots(figsize=(5, 3.8))
ax.plot(years, M_annual, lw=2, marker='o')
ax.set(title='Baseline HANK', xlabel='Year', ylabel='Annual iMPC', ylim=(-0.05, 0.6))
fig.tight_layout()
fig.savefig(f'{folder}/iMPC_annual.pdf')

"""
Present value weights in a row of M
"""

date = 3
dates = np.arange(10)
weights = M_HANK*(1 + ss.r)**np.arange(par.T)[None, :]

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(dates, weights[date, :10], lw=2, label='HANK')
ax.plot(dates, np.full(10, 1 - beta_RANK), lw=2, label='RANK')
ax.axvline(date, color='black', lw=0.8)
ax.set(xlabel='Year s', title=f'Weight of income at date s in consumption at date {date}')
ax.legend()
fig.tight_layout()
fig.savefig(f'{folder}/M_row.pdf')

"""
Fiscal shock and rounds of spending
"""

T_irf = 21
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(100*Y[:T_irf], lw=2, label='dY')
axes[0].plot(100*G[:T_irf], lw=2, label='dG')
axes[0].plot(100*C[:T_irf], lw=2, label='dC')
axes[0].set(title='Output and demand', xlabel='Years', ylabel='% of ss Y')
axes[0].legend()
axes[1].plot(100*G[:T_irf], lw=2, label='dG')
axes[1].plot(100*T[:T_irf], lw=2, label='dT')
axes[1].plot(100*B[:T_irf], lw=2, label='dB')
axes[1].set(title='Government', xlabel='Years')
axes[1].legend()
fig.tight_layout()
fig.savefig(f'{folder}/fiscal_irf.pdf')

fig, ax = plt.subplots(figsize=(7, 3.6))

deficit = G - T
round_k = deficit.copy()
sum_rounds = np.zeros(par.T)

for k in range(1, 41):

    round_k = M_HANK@round_k
    sum_rounds += round_k

    if k in [1, 2, 5, 40]:
        ax.plot(quarters, 100*sum_rounds[:T_max], lw=2, label=f'{k} round' + ('s' if k > 1 else ''))

ax.plot(quarters, 100*C[:T_max], lw=2, ls='--', color='black', label='Full solution')
ax.set(xlabel='Years', ylabel='dC, % of ss Y')
ax.legend()
fig.tight_layout()
fig.savefig(f'{folder}/rounds.pdf')

"""
Deficit-financed tax cut and self-financing
"""

rho_B = 0.90
tax_cut = 0.01*0.9**(4*np.arange(par.T))

debt = np.zeros(par.T)

for t in range(par.T):
    debt[t] = rho_B*debt[t-1] + tax_cut[t] if t > 0 else tax_cut[t]

debt_lag = np.append(0.0, debt[:-1])
T_tax_cut = (1 + ss.r)*debt_lag - debt

identity = np.eye(par.T)
Y_HANK = np.linalg.solve(identity - M_HANK, -M_HANK@T_tax_cut)
Y_TANK = -lambda_TANK/(1 - lambda_TANK)*T_tax_cut

Y_G_check = np.linalg.solve(identity - M_HANK, G - M_HANK@T)
error_check = np.abs(Y_G_check - model.IRF['Y'][:par.T]).max()

fig, ax = plt.subplots(figsize=(7, 3.6))
T_tax = 11
ax.plot(100*Y_HANK[:T_tax], lw=2, label='HANK')
ax.plot(100*Y_TANK[:T_tax], lw=2, label='TANK')
ax.plot(np.zeros(T_tax), lw=2, label='RANK')
ax.set(xlabel='Years', ylabel='dY, % of ss Y')
ax.legend()
fig.tight_layout()
fig.savefig(f'{folder}/tax_cut.pdf')

Y_self = np.linalg.solve(identity - (1 - ss.tau)*M_HANK, G)
revenue_ratio = (q@(ss.tau*Y_self))/(q@G)

debt_self = np.zeros(par.T)

for t in range(par.T):
    debt_lag_self = debt_self[t-1] if t > 0 else 0.0
    debt_self[t] = (1 + ss.r)*debt_lag_self + G[t] - ss.tau*Y_self[t]

T_self = 26
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(100*Y_self[:T_self], lw=2, label='dY')
ax.plot(100*G[:T_self], lw=2, label='dG')
ax.plot(100*ss.tau*Y_self[:T_self], lw=2, label='New tax revenue')
ax.plot(100*debt_self[:T_self], lw=2, label='dB')
ax.set(xlabel='Years', ylabel='% of ss Y')
ax.legend()
fig.tight_layout()
fig.savefig(f'{folder}/self_financing.pdf')

"""
Multipliers and debt financing
"""

rhos_B = [0.0, 0.25, 0.5, 0.75, 0.9]
impact = []
cumulative = []

for rho_B in rhos_B:

    debt = np.zeros(par.T)

    for t in range(par.T):
        debt[t] = rho_B*(debt[t-1] + G[t]) if t > 0 else rho_B*G[t]

    debt_lag = np.append(0.0, debt[:-1])
    T_rho = G + (1 + ss.r)*debt_lag - debt
    Y_rho = np.linalg.solve(identity - M_HANK, G - M_HANK@T_rho)

    impact.append(Y_rho[0]/G[0])
    cumulative.append((q@Y_rho)/(q@G))
    print(f'rho_B = {rho_B}: share = {ss.tau*cumulative[-1]:.3f}, PV dtau/PV dG = {(q@(T_rho - ss.tau*Y_rho))/(q@G):.3f}')

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(rhos_B, impact, lw=2, marker='o', label='Impact')
ax.plot(rhos_B, cumulative, lw=2, marker='o', label='Cumulative')
ax.set(xlabel='Persistence of debt', ylabel='Multiplier')
ax.legend()
fig.tight_layout()
fig.savefig(f'{folder}/multipliers_rho_B.pdf')

dates = [5, 10, 20, 30, 40]
print(f'dY/dB at {dates} = {np.round(Y_rho[dates]/debt[dates], 4)}, dG/dB = {np.round(G[dates]/debt[dates], 4)}')

print(f'lambda_TANK = {lambda_TANK:.3f}')
print(f'error_check = {error_check:.2e}')
print(f'tau = {ss.tau:.3f}, revenue_ratio = {revenue_ratio:.3f}')
print(f'Y_HANK[0, 2, 5, 10] = {100*Y_HANK[[0, 2, 5, 10]].round(5)}, Y_TANK = {100*Y_TANK[[0, 2, 5, 10]].round(5)}')
print(f'Y_self[0, 2, 5, 10, 25, 60] = {100*Y_self[[0, 2, 5, 10, 25, 60]].round(5)}, debt_self peak = {100*debt_self.max():.3f}, debt_self[60, -1] = {100*debt_self[[60, -1]].round(5)}')
print(f'PV Y HANK = {100*(q@Y_HANK):.3f}, PV Y TANK = {100*(q@Y_TANK):.3f}')
print(f'M_annual = {M_annual.round(3)}')
print(f'impact = {np.round(impact, 3)}')
print(f'cumulative = {np.round(cumulative, 3)}')

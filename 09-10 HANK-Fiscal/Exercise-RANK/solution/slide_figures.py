import numpy as np
import matplotlib.pyplot as plt

import RANKModel

"""
Figures for the RANK recap in the slides (run from this folder)
"""

folder = '../../tex/figs'
T_max = 10
quarters = np.arange(T_max)

plt.rcParams.update({'axes.grid': True, 'grid.color': 'black', 'grid.alpha': 0.25, 'grid.linestyle': '--'})

model = RANKModel.NKModelClass(name='baseline')
model.find_ss()
model.compute_jacs()

"""
IRFs to a government spending and a monetary policy shock
"""

for shock, filename, varnames in [('G', 'NK_G', ['Y', 'C', 'pi', 'r']), ('eps_i', 'NK_monpol', ['Y', 'pi', 'i', 'r'])]:

    model.find_transition_path(shocks=[shock], do_print=False, do_end_check=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, varname in zip(axes.flat, varnames):

        ax.plot(quarters, 100*(getattr(model.path, varname)[:T_max, 0] - getattr(model.ss, varname)), lw=2)
        ax.set(title=varname, xlabel='Years', ylabel='Change to ss')

    fig.tight_layout()
    fig.savefig(f'{folder}/{filename}.pdf')

"""
Slope of the wage Phillips curve
"""

fig, ax = plt.subplots()

for kappa in [0.1, 1.0, 10.0, 100.0, 1000.0]:

    model_kappa = model.copy()
    model_kappa.par.kappa = kappa
    model_kappa.compute_jacs()
    model_kappa.find_transition_path(shocks=['eps_i'], do_print=False, do_end_check=False)

    ax.plot(quarters, 100*(model_kappa.path.Y[:T_max, 0] - model_kappa.ss.Y), lw=2, label=f'kappa = {kappa}')

ax.set(xlabel='Years', ylabel='% change in Y')
ax.legend()
fig.savefig(f'{folder}/monpol_NKWC_slope.pdf')

"""
Sticky wages vs sticky prices
"""

# with sticky prices Y and pi are the same to first order, the real wage is on the labor supply curve
model.find_transition_path(shocks=['eps_i'], do_print=False, do_end_check=False)

par, ss, path = model.par, model.ss, model.path
Y, pi, C, L = path.Y[:T_max, 0], path.pi[:T_max, 0], path.C[:T_max, 0], path.L[:T_max, 0]

w_sticky_prices = par.varphi*L**par.nu*C**par.sigma
w_sticky_prices_ss = par.varphi*ss.L**par.nu*ss.C**par.sigma

markup_sticky_prices = 1/w_sticky_prices
markup_sticky_prices_ss = 1/w_sticky_prices_ss

profits_sticky_prices = Y - w_sticky_prices*L
profits_sticky_prices_ss = ss.Y - w_sticky_prices_ss*ss.L

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(quarters, 100*(Y - ss.Y), lw=2, label='Sticky prices')
axes[0, 0].plot(quarters, 100*(Y - ss.Y), lw=2, ls='--', label='Sticky wages')
axes[0, 0].set(title='Y')

axes[0, 1].plot(quarters, 100*pi, lw=2)
axes[0, 1].plot(quarters, 100*pi, lw=2, ls='--')
axes[0, 1].set(title='pi')

axes[1, 0].plot(quarters, 100*(markup_sticky_prices/markup_sticky_prices_ss - 1), lw=2)
axes[1, 0].plot(quarters, np.zeros(T_max), lw=2)
axes[1, 0].set(title='markup')

axes[1, 1].plot(quarters, 100*(profits_sticky_prices - profits_sticky_prices_ss), lw=2)
axes[1, 1].plot(quarters, np.zeros(T_max), lw=2)
axes[1, 1].set(title='profits')

for ax in axes.flat:
    ax.set(xlabel='Years', ylabel='Change to ss')

axes[0, 0].legend()
fig.tight_layout()
fig.savefig(f'{folder}/monpol_irf_sticky_w_p.pdf')

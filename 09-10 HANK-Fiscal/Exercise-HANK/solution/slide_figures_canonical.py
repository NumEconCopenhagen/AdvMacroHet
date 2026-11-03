import numpy as np
import matplotlib.pyplot as plt

import HANKModel

"""
Figures for the Generalized IKC slides (run from this folder)
"""

folder = '../../tex/figs'
T_max = 13
years = np.arange(T_max)

plt.rcParams.update({'axes.grid': True, 'grid.color': 'black', 'grid.alpha': 0.25, 'grid.linestyle': '--'})

model = HANKModel.HANKModelClass(name='canonical',par={'model_type':'canonical'})
model.find_ss()
model.compute_jacs()

par, ss = model.par, model.ss
q = (1 + ss.r)**(-np.arange(par.T))

"""
Government spending shock with real bonds: real rate rule vs Taylor rule
"""

par.nominal_bonds = False

fig, axes = plt.subplots(2, 2, figsize=(7, 4.4))

for taylor, label in [(False, 'Real rate rule'), (True, 'Taylor rule')]:

    par.taylor = taylor
    model.compute_jacs(skip_hh=True)
    model.find_IRFs(shocks=['G'], do_print=False)

    Y, G = model.IRF['Y'][:par.T], model.IRF['G'][:par.T]
    print(f'{label}: impact = {Y[0]/G[0]:.3f}, cumulative = {(q@Y)/(q@G):.3f}')

    for ax, (varname, title) in zip(axes.flat, [('Y', 'Output'), ('C_hh', 'Consumption'), ('pi', 'Inflation'), ('r', 'Real rate')]):

        ax.plot(years, 100*model.IRF[varname][:T_max], lw=2, label=label)
        ax.set(title=f'{title}, % of ss Y' if varname in ['Y', 'C_hh'] else f'{title}, pp', xlabel='Years')

axes[0, 0].legend(fontsize=9)
fig.tight_layout()
fig.savefig(f'{folder}/HANK_G_taylor.pdf')

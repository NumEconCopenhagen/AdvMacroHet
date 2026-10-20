"""Buffer-stock policy functions over a wide wealth range, lecture 4 HANC calibration."""

import os

import numpy as np
import matplotlib.pyplot as plt

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

""" Settings """

def settings():
    """Lecture 4 HANC parameters and the steady-state prices they imply (quarterly)."""

    par = {
        'sigma': 1.0,
        'beta': 0.9810226868872732,
        'rho_z': 0.9,
        'sigma_psi': 0.5,
        'Nz': 7,
        'a_max': 10_000.0,
        'Na': 500,
        'tol': 1e-10,
        'max_iter': 100_000,
    }

    # a. prices from the steady state: K = 16, Y = 1, labor share 2/3
    alpha, delta, Gamma, K = 1/3, 0.008333333333333331, 0.39685026299204984, 16.0
    par['r'] = alpha*Gamma*K**(alpha-1) - delta
    par['w'] = (1-alpha)*Gamma*K**alpha
    return par

""" Solve """

def solve_policies(par):
    """Solve the household problem by EGM until the consumption function converges."""

    a_grid = equilogspace(0.0, par['a_max'], par['Na'])
    z_grid, z_trans, _, _, _ = log_rouwenhorst(par['rho_z'], par['sigma_psi'], par['Nz'])
    r, w = par['r'], par['w']

    c = (1+r)*a_grid[np.newaxis, :] + w*z_grid[:, np.newaxis]
    a = np.zeros_like(c)
    for it in range(par['max_iter']):

        # a. expectation step
        vbeg_a = z_trans @ ((1+r)*c**(-par['sigma']))

        # b. EGM step
        c_old = c.copy()
        for i_z in range(par['Nz']):
            c_endo = (par['beta']*vbeg_a[i_z])**(-1/par['sigma'])
            m_endo = c_endo + a_grid
            m = (1+r)*a_grid + w*z_grid[i_z]
            a[i_z] = np.fmax(np.interp(m, m_endo, a_grid), 0.0)
            c[i_z] = m - a[i_z]

        if np.max(np.abs(c - c_old)) < par['tol']:
            break

    assert it < par['max_iter'] - 1, 'household problem did not converge'
    return a_grid, z_grid, a, c

def target_wealth(a_grid, a_policy):
    """Wealth level above which a household with this productivity dissaves (0 if it always does)."""

    net = a_policy - a_grid
    above = np.where(net[1:] < 0)[0]
    if net[1] < 0:
        return 0.0
    i = above[0] + 1
    return np.interp(0.0, [net[i], net[i-1]], [a_grid[i], a_grid[i-1]])

""" Figure """

def plot_policies(a_grid, z_grid, a, path):
    """Net saving by productivity, up to beyond the top type's target wealth."""

    colors = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#1c5cab', '#0d366b']
    a_star = target_wealth(a_grid, a[-1])
    x_max = 1.3*a_star
    I = a_grid <= x_max

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i_z, color in enumerate(colors):
        ax.plot(a_grid[I], a[i_z, I] - a_grid[I], color=color, lw=2, label=f'$z$ = {z_grid[i_z]:.2f}')

    # a. the top type's target wealth
    ax.axhline(0, color='#3a3a38', lw=0.8)
    ax.axvline(a_star, color='#3a3a38', lw=1, ls='--')
    ax.annotate(f'most productive stop\nsaving at $a \\approx {a_star:.0f}$',
                xy=(a_star, 0), xytext=(0.42*a_star, 0.55*ax.get_ylim()[1]),
                fontsize=9, color='#3a3a38', arrowprops=dict(arrowstyle='->', color='#3a3a38'))

    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('savings change, $a_t - a_{t-1}$')
    ax.set_xlim(0, x_max)
    ax.grid(color='#e4e3df', lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc='lower left', ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return a_star

def main():
    """Solve and save the figure next to the old one in tex/figs_papers."""

    par = settings()
    a_grid, z_grid, a, c = solve_policies(par)
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, 'tex', 'figs', 'policy_functions_wide.pdf')
    a_star = plot_policies(a_grid, z_grid, a, path)
    print(f'r = {par["r"]:.4f}, w = {par["w"]:.4f}, beta(1+r) = {par["beta"]*(1+par["r"]):.4f}')
    print('target wealth by z:', [round(target_wealth(a_grid, a[i]), 1) for i in range(par['Nz'])])
    print(f'top type target wealth: {a_star:.1f}')

if __name__ == '__main__':
    main()

""" numerical tools built in 01_tools.ipynb and reused in the later notebooks """

import numpy as np

""" grids """

def equilogspace(x_min,x_max,n):
    """ grid from x_min to x_max, (close to) equidistant in logs """

    pivot = np.abs(x_min) + 0.25
    grid = np.geomspace(x_min+pivot,x_max+pivot,n) - pivot
    grid[0] = x_min

    return grid

""" interpolation """

def linear_interpolate(G,F,x):
    """ linear interpolation (and extrapolation) of f at the single point x """

    assert len(G) == len(F)
    n = len(G)

    # a. find index in known points
    if x < G[1]:

        i = 0

    elif x > G[-2]:

        i = n-2

    else:

        i = 0
        while x >= G[i+1] and i < n-1:
            i += 1

    # b. interpolate
    slope = (F[i+1]-F[i])/(G[i+1]-G[i])

    return F[i] + slope*(x-G[i])

def linear_interpolate_vec(G,F,x):
    """ linear interpolation (and extrapolation) of f at every point in the array x """

    # a. locate each x between two known points
    i = np.searchsorted(G,x,side='right') - 1
    i = np.clip(i,0,G.size-2)

    # b. interpolate
    slope = (F[i+1]-F[i])/(G[i+1]-G[i])

    return F[i] + slope*(x-G[i])

""" maximization """

def golden_section_search(objective,lower,upper,args=(),n_iter=50):
    """ maximize a concave objective on [lower,upper], vectorized over the bounds """

    inv_phi = (np.sqrt(5)-1)/2

    a = np.asarray(lower,dtype=float).copy()
    b = np.asarray(upper,dtype=float).copy()

    for _ in range(n_iter):

        # a. two interior points
        c = b - inv_phi*(b-a)
        d = a + inv_phi*(b-a)

        # b. keep the bracket containing the maximum
        keep_lower = objective(c,*args) > objective(d,*args)
        b = np.where(keep_lower,d,b)
        a = np.where(keep_lower,a,c)

    return (a+b)/2

""" markov chains """

def find_ergodic(z_trans):
    """ ergodic distribution of a markov transition matrix """

    eigenvalues,eigenvectors = np.linalg.eig(z_trans.T)
    i_unit = np.argmin(np.abs(eigenvalues-1.0))
    ergodic = np.real(eigenvectors[:,i_unit])

    return ergodic/np.sum(ergodic)

def rouwenhorst(rho,sigma,n):
    """ discretize log z_t = rho*log z_{t-1} + psi_t into n states with E[z] = 1 """

    # a. transition matrix by recursion from n = 2
    p = (1+rho)/2
    z_trans = np.array([[p,1-p],[1-p,p]])

    for i in range(3,n+1):

        z_trans_small = z_trans
        z_trans = np.zeros((i,i))
        z_trans[:-1,:-1] += p*z_trans_small
        z_trans[:-1,1:] += (1-p)*z_trans_small
        z_trans[1:,:-1] += (1-p)*z_trans_small
        z_trans[1:,1:] += p*z_trans_small
        z_trans[1:-1] /= 2

    # b. ergodic distribution
    z_ergodic = find_ergodic(z_trans)

    # c. grid spread to match the unconditional standard deviation
    sigma_z = sigma/np.sqrt(1-rho**2)
    log_z_grid = np.linspace(-sigma_z*np.sqrt(n-1),sigma_z*np.sqrt(n-1),n)

    # d. normalize to mean one
    z_grid = np.exp(log_z_grid)
    z_grid /= np.sum(z_ergodic*z_grid)

    return z_grid,z_trans,z_ergodic

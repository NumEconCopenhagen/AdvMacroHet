from math import erf
import numpy as np
from numba import njit

@njit
def norm_cdf(x):
    """ Approximate normal CDF using error function. """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

@njit
def tauchen_trans_nb(grid,mu,rho,sigma):

    n = grid.size
    step = grid[1]-grid[0]

    trans = np.zeros((n,n))

    for j in range(n):
        
        trans[j,0] = norm_cdf((grid[0]-mu-rho*grid[j]+step/2)/sigma)
        trans[j,-1] = 1-norm_cdf((grid[-1]-mu-rho*grid[j]-step/2)/sigma)

        for k in range(1,n - 1):
            trans[j,k] = norm_cdf((grid[k]-mu-rho*grid[j]+step/2)/sigma) - \
                         norm_cdf((grid[k]-mu-rho*grid[j]-step/2) / sigma)
    
    return trans

@njit
def tauchen_nb(mu,rho,sigma,m=4.,n=7,upsilon=0.):
    grid = np.zeros(n)


    # a. grid
    std_grid = np.sqrt(sigma**2/(1-rho**2))  # Unconditional standard deviation
    
    grid[0] = mu/(1-rho) - m*std_grid
    grid[-1] = mu/(1-rho) + m*std_grid

    step = (grid[-1]-grid[0])/(n-1)
    for i in range(1,n-1):
        grid[i] = grid[i-1] + step

    # b. transition matrix
    sigma_new =  sigma + upsilon
    trans = tauchen_trans_nb(grid,mu,rho,sigma_new)

    # c. ergodic distribution
    eigvals, eigvecs = np.linalg.eig(trans.T)
    ergodic = eigvecs[:,np.isclose(eigvals,1)].flatten().real
    ergodic = ergodic / np.sum(ergodic)

    return grid, trans, ergodic

@njit
def log_tauchen_nb(rho,sigma,m=4.5,n=7,upsilon=0.):
    
    # a. standard
    log_grid,trans,ergodic = tauchen_nb(0.0,rho,sigma,m,n,upsilon)
    
    # b. take exp and ensure exact mean of one
    grid = np.exp(log_grid)
    grid /= np.sum(ergodic*grid)

    return log_grid,grid,trans,ergodic
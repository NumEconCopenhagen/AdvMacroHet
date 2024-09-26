import time
import numpy as np
import numba as nb

import quantecon as qe

from EconModel import EconModelClass, jit

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, find_ergodic, choice
from consav.quadrature import log_normal_gauss_hermite
from consav.linear_interp import binary_search, interp_1d
from consav.misc import elapsed

class ConSavModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # preferences
        par.beta = 0.96 # discount factor
        par.sigma = 0.99 # CRRA coefficient
        par.Nbeta = 1  # Number of betas 

        # income
        par.w = 1.0 # wage level
        
        par.rho_zt = 0.96 # AR(1) parameter
        par.sigma_psi = 0.15 # std. of persistent shock
        par.Nzt = 7 # number of grid points for zt
        
        par.sigma_xi = np.nan # std. of transitory shock
        par.Nxi = 1 # number of grid points for xi

        # saving
        par.r = 0.01 # interest rate
        par.b = 0. # borrowing constraint relative to wage

        # grid
        par.a_max = 300.0 # maximum point in grid
        par.Na = 300 # number of grid points       

        # simulation
        par.simT = 500 # number of periods
        par.simN = 100_000 # number of individuals (mc)

        # tolerances
        par.max_iter_solve = 10_000 # maximum number of iterations
        par.max_iter_simulate = 10_000 # maximum number of iterations
        par.tol_solve = 1e-8 # tolerance when solving
        par.tol_simulate = 1e-8 # tolerance when simulating

    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. transition matrix
        
        # persistent
        _out = log_rouwenhorst(par.rho_zt,par.sigma_psi,par.Nzt)
        par.zt_grid,par.zt_trans,par.zt_ergodic,par.zt_trans_cumsum,par.zt_ergodic_cumsum = _out
        par.Nz = par.Nzt
        par.z_grid = par.zt_grid
        par.z_trans = par.zt_trans
        par.z_trans_cumsum = np.cumsum(par.z_trans,axis=1)
        par.z_ergodic = find_ergodic(par.z_trans)
        par.z_ergodic_cumsum = np.cumsum(par.z_ergodic)
        par.z_trans_T = par.z_trans.T

        # b. beta grid 
        par.beta_grid = np.zeros(par.Nbeta)

        # c. asset grid
        assert par.b <= 0.0, f'{par.b = :.1f} > 0, should be non-positive'
        b_min = -par.z_grid.min()/par.r
        if par.b < b_min:
            print(f'parameter changed: {par.b = :.1f} -> {b_min = :.1f}') 
            par.b = b_min + 1e-8

        par.a_grid = par.w*equilogspace(par.b,par.a_max,par.Na)

        # c. solution arrays
        sol.c = np.zeros((par.Nbeta,par.Nz,par.Na))
        sol.a = np.zeros((par.Nbeta,par.Nz,par.Na))
        sol.vbeg = np.zeros((par.Nbeta,par.Nz,par.Na))

        # hist
        sol.pol_indices = np.zeros((par.Nbeta,par.Nz,par.Na),dtype=np.int_)
        sol.pol_weights = np.zeros((par.Nbeta,par.Nz,par.Na))

        # d. simulation arrays

        # hist
        sim.Dbeg = np.zeros((sol.a.shape))
        sim.D = np.zeros((sol.a.shape))
        sim.Dbeg_ = np.zeros(sol.a.shape)

    
    def solve(self,do_print=True):
        """ solve model using value function iteration or egm """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            # time loop
            it = 0
            while True:
                
                t0_it = time.time()

                # a. next-period value function
                if it == 0: # guess on consuming everything
                    m_plus= np.zeros_like(self.sol.a)
                    m_plus[:] = (1+par.r)*par.a_grid[np.newaxis,np.newaxis,:] + par.w*par.z_grid[np.newaxis,:,np.newaxis]
                    c_plus_max = m_plus - par.w*par.b
                    c_plus = 0.99*c_plus_max # arbitary factor
                    v_plus = c_plus**(1-par.sigma)/(1-par.sigma)
                    vbeg_plus = par.z_trans@v_plus

                else:

                    vbeg_plus = sol.vbeg.copy()
                    c_plus = sol.c.copy()

                # b. solve this period
                solve_hh_backwards_egm(par,c_plus,sol.c,sol.a)
                max_abs_diff = np.max(np.abs(sol.c-c_plus))

                # c. check convergence
                converged = max_abs_diff < par.tol_solve
                
                # d. break
                if do_print and (converged or it < 10 or it%100 == 0):
                    print(f'iteration {it:4d} solved in {elapsed(t0_it):10s}',end='')              
                    print(f' [max abs. diff. {max_abs_diff:5.2e}]')

                if converged: break

                it += 1
                if it > par.max_iter_solve: raise ValueError('too many iterations in solve()')
        
        if do_print: print(f'model solved in {elapsed(t0)}')              

    def prepare_simulate(self,do_print=True):
        """ prepare simulation """

        t0 = time.time()

        par = self.par
        sim = self.sim

        sim.Dbeg[:] = 0. 
        sim.Dbeg_[:] = 0. 

        sim.Dbeg[:,:,0] = par.z_ergodic[None,:]/par.Nbeta 
        sim.Dbeg_[:,:,0] = par.z_ergodic[None,:]/par.Nbeta 

        if do_print: print(f'model prepared for simulation in {time.time()-t0:.1f} secs')


    def simulate(self,do_print=True, do_prepare=True):
        """ simulate model """

        t0 = time.time()

        if do_prepare : self.prepare_simulate( do_print=do_print)

        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            Dbeg = sim.Dbeg_
            D = sim.D

            # a. prepare
            find_i_and_w(par,sol)

            # b. iterate
            it = 0 
            while True:

                Dbeg_old = Dbeg.copy()
                simulate_hh_forwards_stochastic(par,Dbeg,D)
                simulate_hh_forwards_choice(par,sol,D,Dbeg)

                max_abs_diff = np.max(np.abs(Dbeg-Dbeg_old))
                if max_abs_diff < par.tol_simulate: 
                    Dbeg[:,:] = Dbeg_old
                    break

                it += 1
                if it > par.max_iter_simulate: raise ValueError('too many iterations in simulate()')

        if do_print: 
            print(f'model simulated in {elapsed(t0)} [{it} iterations]')


##################
# solution - egm #
##################

@nb.njit(parallel=True)
def solve_hh_backwards_egm(par,c_plus,c,a):
    """ solve backwards with c_plus from previous iteration """

    for i_beta in range(par.Nbeta):
        for i_z in nb.prange(par.Nz):

            # a. post-decision marginal value of cash
            q_vec = np.zeros(par.Na)
            for i_z_plus in range(par.Nz):
                q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_beta,i_z_plus,:]**(-par.sigma)
            
            # b. implied consumption function
            c_vec = (par.beta_grid[i_beta]*(1+par.r)*q_vec)**(-1.0/par.sigma)
            m_vec = par.a_grid+c_vec

            # c. interpolate from (m,c) to (a_lag,c)
            for i_a_lag in range(par.Na):
                
                m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
                
                if m <= m_vec[0]: # constrained (lower m than choice with a = 0)
                    c[i_beta,i_z,i_a_lag] = m - par.b*par.w
                    a[i_beta,i_z,i_a_lag] = par.b*par.w
                else: # unconstrained
                    c[i_beta,i_z,i_a_lag] = interp_1d(m_vec,c_vec,m) 
                    a[i_beta,i_z,i_a_lag] = m-c[i_beta,i_z,i_a_lag] 

##########################
# simulation - histogram #
##########################

@nb.njit(parallel=True) 
def find_i_and_w(par,sol):
    """ find pol_indices and pol_weights for simulation """

    i = sol.pol_indices
    w = sol.pol_weights

    for i_beta in nb.prange(par.Nbeta):
        for i_z in nb.prange(par.Nz):
            for i_a_lag in nb.prange(par.Na):
                
                # a. policy
                a_ = sol.a[i_beta,i_z,i_a_lag]

                # b. find i_ such a_grid[i_] <= a_ < a_grid[i_+1]
                i_ = i[i_beta,i_z,i_a_lag] = binary_search(0,par.a_grid.size,par.a_grid,a_) 

                # c. weight
                w[i_beta,i_z,i_a_lag] = (par.a_grid[i_+1] - a_) / (par.a_grid[i_+1] - par.a_grid[i_])

                # d. bound simulation
                w[i_beta,i_z,i_a_lag] = np.fmin(w[i_beta,i_z,i_a_lag],1.0)
                w[i_beta,i_z,i_a_lag] = np.fmax(w[i_beta,i_z,i_a_lag],0.0)

@nb.njit
def simulate_hh_forwards_stochastic(par,Dbeg,D):
    for i_beta in range(par.Nbeta):
        D[i_beta,:] = par.z_trans_T@Dbeg[i_beta]

@nb.njit(parallel=True)   
def simulate_hh_forwards_choice(par,sol,D,Dbeg_plus):
    """ simulate choice transition """

    for i_beta in nb.prange(par.Nbeta):
        for i_z in nb.prange(par.Nz):
        
            Dbeg_plus[i_beta,i_z,:] = 0.0

            for i_a_lag in range(par.Na):
                
                # i. from
                D_ = D[i_beta,i_z,i_a_lag]
                if D_ <= 1e-12: continue

                # ii. to
                i_a = sol.pol_indices[i_beta,i_z,i_a_lag]            
                w = sol.pol_weights[i_beta,i_z,i_a_lag]
                Dbeg_plus[i_beta,i_z,i_a] += D_*w
                Dbeg_plus[i_beta,i_z,i_a+1] += D_*(1.0-w)
import time
import numpy as np
import numba as nb

import quantecon as qe

from EconModel import EconModelClass, jit

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, choice
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
        par.sigma = 2.0 # CRRA coefficient

        # income
        par.w = 1.0 # wage level
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of shock
        par.Nz = 7 # number of grid points

        # saving
        par.r = 0.02 # interest rate

        # grid
        par.a_max = 100.0 # maximum point in grid
        par.Na = 500 # number of grid points       

        # simulation
        par.simT = 500 # number of periods
        par.simN = 100_000 # number of individuals (mc)

        # tolerances
        par.max_iter_solve = 10_000 # maximum number of iteration
        par.tol_solve = 1e-8 # tolerance

    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. asset grid
        par.a_grid = equilogspace(0.0,par.w*par.a_max,par.Na)
        
        # b. productivity grid and transition matrix
        _out = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)
        par.z_grid,par.z_trans,par.z_ergodic,par.z_trans_cumsum,par.z_ergodic_cumsum = _out

        # c. solution arrays
        sol.c = np.zeros((par.Nz,par.Na))
        sol.a = np.zeros((par.Nz,par.Na))
        sol.vbeg = np.zeros((par.Nz,par.Na))

        # hist
        sol.pol_indices = np.zeros((par.Nz,par.Na),dtype=np.int_)
        sol.pol_weights = np.zeros((par.Nz,par.Na))

        # d. simulation arrays

        # mc
        sim.a_ini = np.zeros((par.simN,))
        sim.p_z_ini = np.zeros((par.simN,))
        sim.c = np.zeros((par.simT,par.simN))
        sim.a = np.zeros((par.simT,par.simN))
        sim.p_z = np.zeros((par.simT,par.simN))
        sim.i_z = np.zeros((par.simT,par.simN),dtype=np.int_)

        # hist
        sim.Dbeg = np.zeros((par.simT,*sol.a.shape))
        sim.D = np.zeros((par.simT,*sol.a.shape))

    def solve(self,do_print=True,algo='vfi'):
        """ solve model using value function iteration """

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

                    c_plus = m_plus = (1+par.r)*par.a_grid[np.newaxis,:] + par.w*par.z_grid[:,np.newaxis]
                    v_plus = c_plus**(1-par.sigma)/(1-par.sigma)
                    vbeg_plus = par.z_trans@v_plus

                else:

                    vbeg_plus = sol.vbeg.copy()
                    c_plus = sol.c.copy()

                # b. solve this period
                if algo == 'vfi':
                    solve_hh_backwards_vfi(par,vbeg_plus,c_plus,sol.vbeg,sol.c,sol.a)  
                elif algo == 'egm':
                    solve_hh_backwards_egm(par,c_plus,sol.c,sol.a)
                else:
                    raise NotImplementedError

                # c. check convergence
                max_abs_diff = np.max(np.abs(sol.c-c_plus))
                converged = max_abs_diff < par.tol_solve
                
                # d. break
                if do_print and (converged or it < 10 or it%100 == 0):
                    print(f'iteration {it:4d} solved in {elapsed(t0_it)}',end='')              
                    print(f' [max abs. diff. in c {max_abs_diff:5.2e}]')

                if converged: break

                it += 1
                if it > par.max_iter_solve: raise ValueError('too many iterations in solve()')
        
        if do_print: print(f'model solved in {elapsed(t0)}')              

    def prepare_simulate(self,algo='mc',do_print=True):
        """ prepare simulation """

        t0 = time.time()

        par = self.par
        sim = self.sim

        if algo == 'mc':

            sim.a_ini[:] = 0.0
            sim.p_z_ini[:] = np.random.uniform(size=(par.simN,))
            sim.p_z[:,:] = np.random.uniform(size=(par.simT,par.simN))

        elif algo == 'hist':

            sim.Dbeg[0,:,0] = par.z_ergodic

        else:
            
            raise NotImplementedError

        if do_print: print(f'model prepared for simulation in {time.time()-t0:.1f} secs')

    def simulate(self,algo='mc',do_print=True):
        """ simulate model """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            # prepare
            if algo == 'hist':
                find_i_and_w(par,sol,sol.pol_indices,sol.pol_weights)

            # time loop
            for t in range(par.simT):
                
                if algo == 'mc':
                    simulate_forwards_mc(t,par,sim,sol)
                elif algo == 'hist':
                    sim.D[t] = par.z_trans.T@sim.Dbeg[t]
                    if t == par.simT-1: continue
                    simulate_hh_forwards_choice(par,sol,sim.D[t],sim.Dbeg[t+1])
                else:
                    raise NotImplementedError

        if do_print: print(f'model simulated in {time.time()-t0:.1f} secs')


@nb.njit
def value_of_choice(c,par,i_z,m,vbeg_plus):
    """ value of choice for use in vfi """

    # a. utility
    utility = c[0]**(1-par.sigma)/(1-par.sigma)

    # b. end-of-period assets
    a = m - c[0]

    # c. continuation value     
    vbeg_plus_interp = interp_1d(par.a_grid,vbeg_plus[i_z,:],a)

    # d. total value
    value = utility + par.beta*vbeg_plus_interp
    return value

@nb.njit(parallel=True)        
def solve_hh_backwards_vfi(par,vbeg_plus,c_plus,vbeg,c,a):
    """ solve backwards with v_plus from previous iteration """

    v = np.zeros(vbeg_plus.shape)

    # a. solution step
    for i_z in range(par.Nz):
        for i_a_lag in nb.prange(par.Na):

            # i. cash-on-hand
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
            
            # ii. initial consumption and bounds
            c_guess = np.zeros((1,1))
            bounds = np.zeros((1,2))

            c_guess[0] = c_plus[i_z,i_a_lag]
            bounds[0,0] = 1e-8 
            bounds[0,1] = m

            # iii. optimize
            results = qe.optimize.nelder_mead(value_of_choice,
                c_guess, 
                bounds=bounds,
                args=(par,i_z,m,vbeg_plus))

            # iv. save
            c[i_z,i_a_lag] = results.x[0]
            a[i_z,i_a_lag] = m-c[i_z,i_a_lag]
            v[i_z,i_a_lag] = results.fun # convert to maximum

    # b. expectation step
    vbeg[:,:] = par.z_trans@v

@nb.njit(parallel=True)
def simulate_forwards_mc(t,par,sim,sol):
    """ monte carlo simulation of model. """
    
    c = sim.c
    a = sim.a
    i_z = sim.i_z

    for i in nb.prange(par.simN):

        # a. lagged assets
        if t == 0:
            p_z_ini = sim.p_z_ini[i]
            i_z_lag = choice(p_z_ini,par.z_ergodic_cumsum)
            a_lag = sim.a_ini[i]
        else:
            i_z_lag = sim.i_z[t-1,i]
            a_lag = sim.a[t-1,i]

        # b. productivity
        p_z = sim.p_z[t,i]
        i_z_ = i_z[t,i] = choice(p_z,par.z_trans_cumsum[i_z_lag,:])

        # c. consumption
        c[t,i] = interp_1d(par.a_grid,sol.c[i_z_,:],a_lag)

        # d. end-of-period assets
        m = (1+par.r)*a_lag + par.w*par.z_grid[i_z_]
        a[t,i] = m-c[t,i]

@nb.njit(parallel=True)
def solve_hh_backwards_egm(par,c_plus,c,a):
    """ solve backwards with c_plus from previous iteration """

    for i_z in nb.prange(par.Nz):

        # a. post-decision marginal value of cash
        q_vec = np.zeros(par.Na)
        for i_z_plus in range(par.Nz):
            q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_z_plus,:]**(-par.sigma)
        
        # b. implied consumption function
        c_vec = (par.beta*(1+par.r)*q_vec)**(-1.0/par.sigma)
        m_vec = par.a_grid+c_vec

        # c. interpolate from (m,c) to (a_lag,c)
        for i_a_lag in range(par.Na):
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
            c[i_z,i_a_lag] = interp_1d(m_vec,c_vec,m) 
            c[i_z,i_a_lag] = np.fmin(c[i_z,i_a_lag],m) # bound
            a[i_z,i_a_lag] = m-c[i_z,i_a_lag] 

@nb.njit(parallel=True) 
def find_i_and_w(par,sol,i,w):
    """ find pol_indices and pol_weights for simulation """

    for i_z in nb.prange(par.Nz):
        for i_a_lag in nb.prange(par.Na):
            
            # a. policy
            a_ = sol.a[i_z,i_a_lag]

            # b. find i_ such a_grid[i_] <= a_ < a_grid[i_+1]
            i_ = i[i_z,i_a_lag] = binary_search(0,par.a_grid.size,par.a_grid,a_) 

            # c. weight
            w[i_z,i_a_lag] = (par.a_grid[i_+1] - a_) / (par.a_grid[i_+1] - par.a_grid[i_])

            # d. bound simulation
            w[i_z,i_a_lag] = np.fmin(w[i_z,i_a_lag],1.0)
            w[i_z,i_a_lag] = np.fmax(w[i_z,i_a_lag],0.0)

@nb.njit(parallel=True)   
def simulate_hh_forwards_choice(par,sol,D,Dbar_plus):
    """ simulate choice transition """

    for i_z in nb.prange(par.Nz):
    
        Dbar_plus[i_z,:] = 0.0

        for i_a_lag in range(par.Na):
            
            # i. from
            D_ = D[i_z,i_a_lag]

            # ii. to
            i = sol.pol_indices[i_z,i_a_lag]            
            w = sol.pol_weights[i_z,i_a_lag]
            Dbar_plus[i_z,i] += D_*w
            Dbar_plus[i_z,i+1] += D_*(1.0-w)
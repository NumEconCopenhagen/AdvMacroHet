import time
import numpy as np
import numba as nb

import quantecon as qe

from EconModel import EconModelClass, jit

from consav.grids import equilogspace
from consav.markov import choice
from consav.linear_interp import interp_1d
from consav.misc import elapsed

class ConSavModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        # random numbers
        self.rng = np.random.default_rng(1234)

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # preferences
        par.beta = 0.80 # discount factor
        par.sigma = 2.0 # CRRA coefficient

        # income
        par.w = 1.0 # wage level        
        par.pi = 0.50 # prob. of not changing in productivity
        par.Delta = 0.80 # absolute change in productivity (+/-)
        par.chi = np.array([0.0,0.0,0.0]) # deterministic transfer component of income

        # saving
        par.r = 0.0 # interest rate
        par.a_min = -0.10 # borrowing constraint

        # grid
        par.T = 3 # number of periods
        par.a_max = 10.0 # maximum asset value in grid
        par.Na = 500 # number of grid points
        par.Nz = 3 # number of productivity states    

        # simulation
        par.simN = 100_000 # number of individuals to simulate

    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        sim = self.sim
    
        # a. asset grid        
        par.a_grid = equilogspace(par.a_min,par.a_max,par.Na)

        # b. productivity grid
        par.z_grid = np.array([1.0-par.Delta,1.0,1.0+par.Delta])

        # c. transition matrix
        par.z_trans = np.array(
            [[1.0,0.0,0.0],
             [(1-par.pi)/2,par.pi,(1-par.pi)/2],
             [0.0,0.0,1.0]]
            )

        par.z_trans_cumsum = np.cumsum(par.z_trans,axis=1)

        # c. solution arrays
        sol.c = np.zeros((par.T,par.Nz,par.Na))
        sol.m = np.zeros((par.T,par.Nz,par.Na))
        sol.a = np.zeros((par.T,par.Nz,par.Na))
        sol.v = np.zeros((par.T,par.Nz,par.Na))

        # d. simulation
        sim.a_ini = np.zeros((par.simN,))
        sim.c = np.zeros((par.T,par.simN))
        sim.a = np.zeros((par.T,par.simN))
        sim.p_z = np.zeros((par.T,par.simN))
        sim.i_z = np.zeros((par.T,par.simN),dtype=np.int_)
   
    def solve(self,do_print=True,algo='vfi'):
        """ solve model using value function iteration or egm """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            # time loop
            for k in range(par.T):

                t = par.T-k-1
                
                t0_t = time.time()

                # a. solve
                if k == 0:
                    
                    sol.a[t,:,:] = 0.0
                    sol.c[t,:,:] = (1+par.r)*par.a_grid[np.newaxis,:] + par.w*par.z_grid[:,np.newaxis] + par.chi[t]
                    sol.v[t,:,:] = sol.c[t,:,:]**(1-par.sigma)/(1-par.sigma)

                else:

                    if algo == 'vfi':
                        v_plus = sol.v[t+1,:,:]
                        solve_hh_backwards_vfi(par,v_plus,sol.v[t],sol.c[t],sol.a[t],par.chi[t])  
                    elif algo == 'egm':
                        c_plus = sol.c[t+1,:,:]
                        solve_hh_backwards_egm(par,c_plus,sol.c[t],sol.a[t],par.chi[t])
                    else:
                        raise NotImplementedError

                if do_print: print(f'period {t = :2d} solved in {elapsed(t0_t):10s}')
        
        if do_print: print(f'model solved in {elapsed(t0)}')              

    def prepare_simulate(self,do_print=True):
        """ prepare simulation """

        t0 = time.time()

        par = self.par
        sim = self.sim
        rng = self.rng

        sim.a_ini[:] = rng.exponential(scale=1.0,size=(par.simN,))
        sim.p_z[:,:] = rng.uniform(size=(par.T,par.simN))

        if do_print: print(f'model prepared for simulation in {time.time()-t0:.1f} secs')

    def simulate(self,algo='mc',do_print=True):
        """ simulate model """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            for t in range(par.T):               
                simulate_forwards_mc(t,par,sim,sol)
 
        if do_print: print(f'model simulated in {elapsed(t0)} secs')

##################
# solution - vfi #
##################

@nb.njit
def value_of_choice(c,par,i_z,m,v_plus):
    """ value of choice for use in vfi """

    # a. utility
    utility = c**(1-par.sigma)/(1-par.sigma)

    # b. end-of-period assets
    a = m - c

    # c. continuation value
    v_plus_interp = 0.0     
    for i_z_plus in range(3):

        z_trans = par.z_trans[i_z,i_z_plus]
        if np.isclose(z_trans,0.0): continue 
        v_plus_interp += z_trans*interp_1d(par.a_grid,v_plus[i_z_plus,:],a)

    # d. total value
    value = utility + par.beta*v_plus_interp

    return value

@nb.njit(parallel=True)        
def solve_hh_backwards_vfi(par,v_plus,v,c,a,chi):
    """ solve backwards with v_plus from previous iteration """

    # a. solution step
    for i_z in nb.prange(par.Nz):
        for i_a_lag in range(par.Na):

            # i. cash-on-hand and maximum consumption
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z] + chi
            c_max = m - par.a_min

            # ii. optimize
            xf,fval,info = qe.optimize.brent_max(value_of_choice,
                1e-8,c_max,xtol=1e-8,
                args=(par,i_z,m,v_plus))
            
            # iv. save
            c[i_z,i_a_lag] = xf
            a[i_z,i_a_lag] = m-c[i_z,i_a_lag]
            v[i_z,i_a_lag] = fval
            
##################
# solution - egm #
##################

@nb.njit(parallel=True)
def solve_hh_backwards_egm(par,c_plus,c,a,chi):
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
            
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z] + chi
            
            if m <= m_vec[0]: # constrained (lower m than choice with a = a_min)
                c[i_z,i_a_lag] = m - par.a_min
                a[i_z,i_a_lag] = par.a_min
            else: # unconstrained
                c[i_z,i_a_lag] = interp_1d(m_vec,c_vec,m) 
                a[i_z,i_a_lag] = m-c[i_z,i_a_lag] 

############################
# simulation - monte carlo #
############################

@nb.njit(parallel=True)
def simulate_forwards_mc(t,par,sim,sol):
    """ monte carlo simulation of model. """
    
    c = sim.c
    a = sim.a
    i_z = sim.i_z

    for i in nb.prange(par.simN):

        # a. lagged productivity and lagged assets
        if t == 0:
            i_z_lag = 1
            a_lag = sim.a_ini[i]
        else:
            i_z_lag = sim.i_z[t-1,i]
            a_lag = sim.a[t-1,i]

        # b. productivity
        p_z = sim.p_z[t,i]
        i_z_ = i_z[t,i] = choice(p_z,par.z_trans_cumsum[i_z_lag,:])

        # c. consumption
        c[t,i] = interp_1d(par.a_grid,sol.c[t,i_z_,:],a_lag)

        # d. end-of-period assets
        m = (1+par.r)*a_lag + par.w*par.z_grid[i_z_] + par.chi[t]
        a[t,i] = m-c[t,i]
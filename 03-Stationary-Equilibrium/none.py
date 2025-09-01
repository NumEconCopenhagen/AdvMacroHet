# %% [markdown]
# # Stationary Equilibrium

# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - 1. [Setup](#toc1_)    
# - 2. [Solve household problem](#toc2_)    
# - 3. [Find stationary equilibrium](#toc3_)    
#   - 3.1. [Direct approach](#toc3_1_)    
#   - 3.2. [Looking at the stationary equilibrium](#toc3_2_)    
#     - 3.2.1. [Policy functions](#toc3_2_1_)    
#     - 3.2.2. [Distributions](#toc3_2_2_)    
#   - 3.3. [Indirect approach](#toc3_3_)    
# - 4. [Idiosyncratic risk and the steady state interest rate](#toc4_)    
# - 5. [Calibration](#toc5_)    
# - 6. [Extra: Demand vs. supply of capital](#toc6_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=true
# 	anchor=true
# 	flat=false
# 	minLevel=2
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %% [markdown]
# **Contents:**
# 
# 1. Introduces the `GEModelTools`
# 1. Solves and simulates a simple **Heterogenous Agent Neo-Classical (HANC) model**

# %%
#%load_ext autoreload
#%autoreload 2

import time
import pickle
import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt   
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({"axes.grid" : True, "grid.color": "black", "grid.alpha":"0.25", "grid.linestyle": "--"})
plt.rcParams.update({'font.size': 14})

from HANCModel import HANCModelClass

# %% [markdown]
# ## 1. <a id='toc1_'></a>[Setup](#toc0_)

# %%
model = HANCModelClass(name='baseline') # create an instance of the model

# %%
par = model.par
ss = model.ss

# %% [markdown]
# **Pause:** Take a look at `.par` and `.ss`

# %% [markdown]
# **Question I:** Where does the variable names in `.ss` come from?

# %% [markdown]
# **Question II:** What explains the shapes below?

# %%
print(ss.a.shape)
print(ss.D.shape)
print(ss.z_trans.shape)

# %% [markdown]
# ## 2. <a id='toc2_'></a>[Solve household problem](#toc0_)

# %% [markdown]
# Set the steady state values, which matter for the household:

# %%
ss.r = 0.01
ss.w = 1.00

# %% [markdown]
# **Solve the household problem** with `.solve_hh_ss()`:
#     
# 1. Calls `.prepare_hh_ss()`
# 1. Calls `.solve_backwards_hh()` until convergence

# %%
model.solve_hh_ss(do_print=True)

# %%
model.simulate_hh_ss(do_print=True)

# %% [markdown]
# **Aggregate savings:**

# %%
np.sum(ss.a*ss.D)

# %% [markdown]
# ## 3. <a id='toc3_'></a>[Find stationary equilibrium](#toc0_)

# %% [markdown]
# ### 3.1. <a id='toc3_1_'></a>[Direct approach](#toc0_)

# %%
model.find_ss(method='direct',do_print=True)

# %% [markdown]
# **Look at the steady state variables:**

# %%
for varname in model.varlist:
    print(f'{varname:15s}: {ss.__dict__[varname]:.4f}')

# %%
model.info(ss=True)

# %% [markdown]
# ### 3.3. <a id='toc3_3_'></a>[Indirect approach](#toc0_)

# %%
model.find_ss(method='indirect',do_print=True)

# %% [markdown]
# **Question:** What are the pros and cons of the direct and indirect method?

# %% [markdown]
# ### 3.2. <a id='toc3_2_'></a>[Looking at the stationary equilibrium](#toc0_)

# %% [markdown]
# #### 3.2.1. <a id='toc3_2_1_'></a>[Policy functions](#toc0_)

# %%


fig = plt.figure(figsize=(12,4),dpi=100)

I = par.a_grid < 500

# a. consumption
ax = fig.add_subplot(1,2,1)
ax.set_title(f'consumption')

for i_z,z in enumerate(par.z_grid):
    if i_z%3 == 0 or i_z == par.Nz-1:
        ax.plot(par.a_grid[I],ss.c[0,i_z,I],label=f'z = {z:.2f}')

ax.legend(frameon=True)
ax.set_xlabel('savings, $a_{t-1}$')
ax.set_ylabel('consumption, $c_t$')
ax.set_xlim(0,100)
ax.set_ylim(0,5)

# b. saving
ax = fig.add_subplot(1,2,2)
ax.set_title(f'saving')

for i_z,z in enumerate(par.z_grid):
    if i_z%3 == 0 or i_z == par.Nz-1:
        ax.plot(par.a_grid[I],ss.a[0,i_z,I]-par.a_grid[I],label=f'z = {z:.2f}')

ax.set_xlabel('savings, $a_{t-1}$')
ax.set_ylabel('savings change, $a_{t}-a_{t-1}$')
ax.set_xlim(0,100)
ax.set_ylim(-3,1)

fig.tight_layout()
#fig.savefig(f'tex/figs/c_func.pdf')

# %% [markdown]
# #### 3.2.2. <a id='toc3_2_2_'></a>[Distributions](#toc0_)

# %%
fig = plt.figure(figsize=(12,4),dpi=100)

# a. income
ax = fig.add_subplot(1,2,1)
ax.set_title('productivity')
for i_beta,beta in enumerate(par.beta_grid):
    ax.plot(par.z_grid,np.cumsum(np.sum(ss.D[i_beta],axis=1))*par.Nfix,label=f'$\\beta = {beta:.4f}$')

ax.set_xlabel('productivity, $z_{t}$')
ax.set_ylabel('CDF')
ax.legend()

# b. assets
ax = fig.add_subplot(1,2,2)
ax.set_title('savings')
for i_beta in range(par.Nfix):
    ax.plot(np.insert(par.a_grid,0,par.a_grid[0]),np.insert(np.cumsum(np.sum(ss.D[i_beta],axis=0)),0,0.0)*par.Nfix,label=f'$\\beta = {par.beta_grid[i_beta]:.4f}$')
ax.set_xlabel('assets, $a_{t}$')
ax.set_ylabel('CDF')
ax.set_xscale('symlog')

fig.tight_layout()
#fig.savefig('tex/figs/distribution.pdf')

# %% [markdown]
# **Income moments:**

# %%
mean_z = np.sum(ss.D*par.z_grid[:,np.newaxis])
std_z = np.sqrt(np.sum(ss.D*(par.z_grid[np.newaxis,:,np.newaxis]-mean_z)**2))
print(f'mean z: {mean_z:5.2f}')
print(f'std. z: {std_z:5.2f}')

# %% [markdown]
# **Asset moments:**

# %%
# a. prepare
Da = np.sum(ss.D,axis=(0,1))
Da_cs = np.cumsum(Da)
mean_a = np.sum(Da*par.a_grid)
std_a = np.sqrt(np.sum(Da*(par.a_grid-mean_a)**2))

def percentile(par,Da_cs,p):
    
    # a. check first
    if p < Da_cs[0]: return par.a_grid[0]
    
    # b. find with loop
    i = 0
    while True:
        if p > Da_cs[i+1]:
            if i+1 >= par.Na: raise Exception()
            i += 1
            continue
        else:
            w = (p-Da_cs[i])/(Da_cs[i+1]-Da_cs[i])
            diff = par.a_grid[i+1]-par.a_grid[i]
            return par.a_grid[i]+w*diff
        
p25_a = percentile(par,Da_cs,0.25)
p50_a = percentile(par,Da_cs,0.50)
p95_a = percentile(par,Da_cs,0.95)
p99_a = percentile(par,Da_cs,0.99)

# b. print
print(f'mean a: {mean_a:6.3f}')
print(f'p25  a: {p25_a:6.3f}')
print(f'p50  a: {p50_a:6.3f}')
print(f'p95  a: {p95_a:6.3f}')
print(f'p99  a: {p99_a:6.3f}')
print(f'std. a: {std_a:6.3f}')

# %% [markdown]
# **MPC:**

# %%
def calc_MPC(par,ss):
    
    MPC = np.zeros(ss.D.shape)
    dc = (ss.c[:,:,1:]-ss.c[:,:,:-1])
    dm = (1+model.ss.r)*par.a_grid[np.newaxis,np.newaxis,1:]-(1+model.ss.r)*par.a_grid[np.newaxis,np.newaxis,:-1]
    MPC[:,:,:-1] = dc/dm
    MPC[:,:,-1] = MPC[:,:,-1] # assuming constant MPC at end
    mean_MPC = np.sum(MPC*ss.D)
    return mean_MPC

mean_MPC = calc_MPC(par,ss)
print(f'mean MPC: {mean_MPC:.3f}')

# %% [markdown]
# **Question:** What is the correlation between income and savings?

# %% [markdown]
# ## 4. <a id='toc4_'></a>[Idiosyncratic risk and the steady state interest rate](#toc0_)

# %%
print(f'ss.A_hh = ss.K = {ss.A_hh:.2f}')
print(f'ss.r = {ss.r*100:.2f} %')
print('')
      
for sigma_psi in np.linspace(par.sigma_psi,2*par.sigma_psi,5):
    
    print(f'{sigma_psi = :.2f}')

    model_ = model.copy()
    model_.par.sigma_psi = sigma_psi
        
    model_.solve_hh_ss(do_print=False)
    model_.simulate_hh_ss(do_print=False)
    
    A_hh = np.sum(model_.ss.a*model_.ss.D)
    
    print(f'PE {A_hh = :.2f}')
          
    model_.find_ss(method='direct')

    print(f'GE ss.r = {model_.ss.r*100:.2f} %')
    print(f'GE ss.A_hh = ss.K = {model_.ss.A_hh:.2f}')

    print('')

# %% [markdown]
# ## 5. <a id='toc5_'></a>[Calibration](#toc0_)

# %% [markdown]
# Choose `beta_mean` to get chosen average MPC.

# %%
from root_finding import brentq

# %%
def calib_obj(beta_mean,model):
    """ calibration objective """
    
    model.par.beta_mean = beta_mean
    model.find_ss(method='direct')    
    
    mean_MPC = calc_MPC(model.par,model.ss)
    
    return mean_MPC-0.27

# %%
model_calib = model.copy()
brentq(calib_obj,par.beta_mean-0.01,par.beta_mean,args=(model_calib,),do_print=True,varname='beta_mean',funcname='MPC-0.27',xtol=1e-8,rtol=1e-8);

# %%
print(f'ss.r = {model_calib.ss.r*100:.2f} %')
print(f'ss.K = {model_calib.ss.K:.2f}')

# %% [markdown]
# **Question:** What could an alternative be to use a root-finder?

# %% [markdown]
# ## 6. <a id='toc6_'></a>[Extra: Demand vs. supply of capital](#toc0_)

# %%
# allocate
Nr_ss = 7
r_ss_min = 0.007
r_ss_max = 0.013
r_ss_vec = np.linspace(r_ss_min,r_ss_max,Nr_ss)

K_hh_supply = np.zeros(Nr_ss)
K_firm_demand = np.zeros(Nr_ss)

# calculate
for i,r_ss in enumerate(r_ss_vec):
    
    print(f'{r_ss = :7.4f}')
          
    model_ = model.copy()
    model_.ss.r = r_ss
    
    # a. firms
    K_firm_demand[i] = ((r_ss+par.delta)/(par.alpha*ss.Gamma))**(1/(par.alpha-1))
    print(f'K_firm_demand = {K_firm_demand[i]:7.4f}')
    
    w_ss = model_.ss.w = (1.0-par.alpha)*ss.Gamma*(K_firm_demand[i]/ss.L)**par.alpha

    print(f'{w_ss = :7.4f}')

    # b. households
    model_.solve_hh_ss(do_print=True)
    model_.simulate_hh_ss(do_print=True)
    
    K_hh_supply[i] = np.sum(model_.ss.a*model_.ss.D)
    print(f'K_hh_supply = {K_hh_supply[i]:7.4f}')
          
    # c. clearing
    clearing_A = K_hh_supply[i]-K_firm_demand[i]
    print(f'{clearing_A = :7.4f}\n')
    

# %%
fig = plt.figure(figsize=(6,6/1.5),dpi=100)
ax = fig.add_subplot(1,1,1)

ax.axvline(ss.r,color='black')
ax.axhline(ss.K,color='black')

ax.plot(r_ss_vec,K_hh_supply,'-o',label='hh supply')
ax.plot(r_ss_vec,K_firm_demand,'-o',label='firm demand')

ax.set_xlabel('$r_{ss}$')
ax.set_ylabel('$K_{ss}$')
ax.legend(frameon=True);



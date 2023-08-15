import os
import glob
from types import SimpleNamespace
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import norm

#############
# 1. lpirfs #
#############

def lpirfs(data_IRF,shockname,endo,contemp,exo):
    """ compute IRF using lpirfs in R """

    # a. write excel
    data_IRF[endo].to_excel('IRF_endo.xlsx',index=False)
    data_IRF[contemp].to_excel('IRF_contemp.xlsx',index=False)
    data_IRF[exo].to_excel('IRF_exog.xlsx',index=False)
    data_IRF[shockname].to_excel('IRF_shock.xlsx',index=False)    

    # b. run R
    os.system(f'jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 "lpirfs.ipynb" --output lpirfs_temp.ipynb')

    # c. load results
    mean = pd.read_excel('IRF_mean.xlsx').values
    lower = pd.read_excel('IRF_lower.xlsx').values
    upper = pd.read_excel('IRF_upper.xlsx').values

    # d. IRFs
    IRF = {}
    for i,varname in enumerate(endo):
        
        # i. monthly
        IRF[varname] = mean[i,:]
        IRF[(varname,'L')] = lower[i,:]
        IRF[(varname,'U')] = upper[i,:]

    # e. clean up
    os.remove('lpirfs_temp.ipynb')
    for filename in glob.glob('IRF_*.xlsx'):
        os.remove(filename)
    
    return IRF

##########
# 2. slp #
##########

# TODO:
# 1. What are the correct confidence bands?
# 2. Add cross-validation
# 3. How to choose r?


def bspline(x,xl,xr,ndx,bdeg):
    """ construct B-spline """

    dx = (xr-xl)/ndx
    dx = (xr - xl) / ndx
    
    t = xl + dx *np.arange(-bdeg,ndx)
    T = np.tile(t,ndx).reshape((ndx,t.size))
    X = np.repeat(x,t.size).reshape((ndx,t.size))

    P = (X - T) / dx
    B = (T <= X) & (X < (T+dx))
    
    r = np.zeros(t.size,dtype=np.int)
    r[0:-1] = np.arange(1,t.size)
    r[-1] = 0
    
    for k in range(1,bdeg+1):
        B = (P*B + (k+1-P)*B[:,r])/k
        
    return B

def construct_RHS(contemp,exo,Nlags):
    """ construct RHS matrix """

    if not contemp is None:
        assert contemp.ndim == 2
        assert contemp.shape[0] == exo.shape[0]

    assert exo.ndim == 2

    # a. number of timeperiods
    
    # b. number of variables (incl. constant)
    if not contemp is None:
        K_contemp = contemp.shape[1]
        T = contemp.shape[0]
    else:
        T = exo.shape[0]
        K_contemp = 0

    K_exo = exo.shape[1]
    K = 1 +  K_contemp + Nlags*K_exo

    # c. construct RHS
    RHS = np.ones((T,K))

    # constant
    RHS[:,0] = 1 

    # contemporaneous
    if K_contemp > 0:
        RHS[:,1:K_contemp+1] = contemp

    # exogenous
    for lag in range(1,Nlags+1):
        RHS[:lag,K_contemp+1+(lag-1)*K_exo:K_contemp+1+lag*K_exo] = 0.0
        RHS[lag:,K_contemp+1+(lag-1)*K_exo:K_contemp+1+lag*K_exo] = exo[:-lag,:]
    
    # d. restrict to finite
    I = np.isfinite(RHS)
    RHS[~I] = 0
    
    return RHS

def slp(data_IRF,endoname,shockname,contemp,exo,Nlags,H_min,H_max,smooth=False,r=0,lambdaval=0):
    """ estimate IRF with local projection """

    # a. construct RHS

    # unpack
    endo = np.expand_dims(data_IRF[endoname].values,axis=1)
    shock = np.expand_dims(data_IRF[shockname].values,axis=1)
    if len(contemp) > 0:
        contemp = np.vstack([data_IRF[varname].values for varname in contemp]).T
    else: 
        contemp = None
    exo = np.vstack([data_IRF[varname].values for varname in exo]).T

    # construct
    RHS = construct_RHS(contemp,exo,Nlags)
    
    # remove nans
    endo = endo[Nlags:,:]
    shock = shock[Nlags:,:]
    RHS = RHS[Nlags:,:]

    # b. auxiliary computations
    T = endo.size
    L = (H_max+1)*T
    HR = H_max+1-H_min
    NRHS = RHS.shape[1]

    if smooth:
        B = bspline(np.arange(H_min,H_max+1),H_min,H_max+1,H_max+1-H_min,3)
        K = B.shape[1]
    else:
        K = HR

    # c. construct regression representation
    
    # allocate
    idx = np.nan*np.ones((L,2))
    Y = np.nan*np.ones((L,1))
    Xb = np.zeros((L,K))
    Xc = np.zeros((L,HR,NRHS))

    # construct idx, Y, Xb, X
    for t in range(T-H_min):

        # i. idx
        idx_beg = t*HR
        idx_end = (t+1)*HR

        idx[idx_beg:idx_end,0] = t
        idx[idx_beg:idx_end,1] = np.arange(H_min,H_max+1)

        # ii. Y
        y_range = np.arange((t+H_min),np.fmin((t+1+H_max),T)).T
        Y[idx_beg:idx_end] = np.nan
        Y[idx_beg:idx_beg+y_range.size] = endo[y_range]

        # iii. Xb
        if smooth:
            Xb[idx_beg:idx_end,:] = B*shock[t]
        else:
            Xb[idx_beg:idx_end,:] = np.eye(HR)*shock[t]

        # iv. Xc
        for i in range(NRHS):
            Xc[idx_beg:idx_end,:,i] = np.eye(HR)*RHS[t,i]

    # filter Y
    I = np.isfinite(Y)
    Y = Y[I]

    idx = idx[I.ravel(),:]

    # construct X
    X = np.nan*np.ones((Y.size,K+HR*NRHS))

    I = I.reshape((I.size,))
    for i in range(K):
        X[:,i] = Xb[I,i]

    for i in range(NRHS):
        X[:,K+i*HR:K+(i+1)*HR] = Xc[I,:,i]

    IRF = np.zeros((H_max+1,1))

    # d. estimate
    if smooth:

        # i. construct P
        P = np.zeros((X.shape[1],X.shape[1]))
        D = np.eye(K)
        for _ in range(r):
            D = np.diff(D,axis=0)
        
        P[:K,:K] = D.T@D

        # ii. estimate
        theta = np.linalg.solve(X.T@X + lambdaval*P,X.T@Y)
        
        # iii. derive IR
        IRF[H_min:,0] = B@theta[:K]

    else:

        # i. estimate
        theta = np.linalg.solve(X.T@X,X.T@Y)    

        # ii. derive IR
        IRF[H_min:,0] = theta[:K]
    
    # e. save output
    res = SimpleNamespace()

    res.H_min = H_min
    res.H_max = H_max
    res.smooth  = smooth

    res.T = T
    res.HR = HR

    res.K = K
    
    if smooth:
        res.B = B
        res.P = P
        res.lambdaval = lambdaval
    else:        
        res.B = 0
        res.P = np.zeros((X.shape[1],X.shape[1]))
        res.lambdaval = 0
    
    res.idx = idx
    res.Y = Y
    res.X = X
    res.theta = theta
    res.IRF = IRF

    return res

def slp_conf(res,H,lambdaval=0.0,fast=True):
    """ calculate confidence bands """

    # a. point estimate
    XXP = res.X.T@res.X + lambdaval*res.P

    # b. compute NW estimator
    
    # bread
    bread = np.linalg.inv(XXP)

    # meat
    if fast:
        nlag = H+1
    else:
        nlag = H

    T = res.T
    npar = res.theta.size

    if fast:

        weights = (nlag+1-np.arange(0,nlag+1))/(nlag+1)

    else:
        
        weights = np.nan*np.ones(nlag+1)
        weights[0] = 0.5
        weights[1:] = (nlag+1-np.arange(1,nlag+1))/(nlag+1)        

    idx = res.idx
    U = res.Y - res.X@res.theta    
    X = res.X 
    V = np.zeros( (npar,npar) )

    if fast:

        S = X*np.repeat(U,X.shape[1]).reshape(X.shape)
        V = S.T@S

        for i in range(nlag+1):
            Gammai = S[i:T+1,:].T@S[:(T+1-i),:]
            GplusGprime = Gammai+Gammai.T
            V = V + weights[i]*GplusGprime
    
    else:

        for l in range(nlag+1):
            GplusGprime = np.zeros( (npar,npar) )
            for t in range(l,T-res.HR-1):
                S1 = X[ idx[:,0] == t , : ].T @ U[ idx[:,0] == t ]
                S2 = X[ idx[:,0] == t-l , : ].T @ U[ idx[:,0] == t-l ]
                GplusGprime = GplusGprime + S1[:,np.newaxis]@S2[:,np.newaxis].T + S2[:,np.newaxis]@S1[:,np.newaxis].T
            V = V + weights[l] * GplusGprime

    meat = V

    # VC
    VC = bread@meat@bread

    # c. confidence span
    conf = np.nan*np.ones( (res.H_max+1,2) )
    if res.smooth:
        ster = np.sqrt(np.diag(res.B@VC[:res.K,:res.K]@res.B.T))                
        conf[res.H_min:,0] = res.B@res.theta[:res.K] + ster*norm.ppf(0.05)
        conf[res.H_min:,1] = res.B@res.theta[:res.K] + ster*norm.ppf(0.95)
    else:
        ster = np.sqrt(np.diag(VC[:(res.H_max+1-res.H_min),:(res.H_max+1-res.H_min)]))
        conf[res.H_min:,0] = res.theta[:res.K] + ster*norm.ppf(0.05)
        conf[res.H_min:,1] = res.theta[:res.K] + ster*norm.ppf(0.95)

    # d. output
    res.ster = ster
    res.conf = conf

    return res
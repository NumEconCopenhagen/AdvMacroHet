import numpy as np 

def find_ss(model):
    """ find the steady state """

    par = model.par
    ss = model.ss
    
    ss.P = 1. 
    ss.eps_i = 0. 
    ss.i = ss.r = par.i_ss
    ss.beta = 1/(1+ss.r)
    par.betaF = ss.beta
    ss.mc = 1/par.mu
    ss.pi = 0.0 
    ss.Y = ss.C = ss.N = 1.0 
    ss.Z = ss.Y/ss.N 
    ss.w = ss.mc * ss.Z         
    ss.G = 0. 
    ss.B = 0. 
    ss.LT = 0. 
    ss.A = 0. 
    ss.profits = ss.Y - (ss.w*ss.N + par.theta/2 * ss.pi**2 * ss.Y)
    par.vphi = ss.w / ss.N**(par.inv_frisch) / (ss.C**(-par.CRRA))  



import numpy as np

def sort_by_value(x,weights):
    """ flatten, sort by value and normalize the weights """

    x = x.ravel()
    weights = weights.ravel()
    I = np.argsort(x)

    return x[I],weights[I]/np.sum(weights)

def top_shares(x,weights,tops=(0.1,0.01,0.001)):
    """ shares of the total held by the top fractions of the population, and by the bottom 50% """

    x,weights = sort_by_value(x,weights)
    cum_weights = np.cumsum(weights)
    cum_x = np.cumsum(x*weights)/np.sum(x*weights)

    shares = {'bottom 50%':np.interp(0.5,cum_weights,cum_x)}
    for top in tops:
        shares[f'top {100*top:g}%'] = 1.0-np.interp(1.0-top,cum_weights,cum_x)

    return shares

def average_top_wealth(x,weights,top):
    """ average of x among the top fraction of the population """

    share = top_shares(x,weights,tops=(top,))[f'top {100*top:g}%']
    return share*np.sum(x*weights)/np.sum(weights)/top

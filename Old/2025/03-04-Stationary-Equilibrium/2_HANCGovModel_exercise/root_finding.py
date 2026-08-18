def brentq(f,a,b,args=(),xtol=1e-12,rtol=1e-12,max_iter=1_000,
    do_print=False,varname='x',funcname='f'):
    """ brentq root-finder """

    # a. initial evaluation
    fa = f(a,*args)
    fb = f(b,*args)

    if fa*fb > 0:
        raise ValueError(f'f(a) and f(b) must have different signs [{funcname}]')
    elif fa == 0:
        root,converged = a,True
    elif fb == 0:
        root,converged = b,True
    else: 
        root,converged = None,False

    if converged:
        if do_print: print(f'{varname} = {root:12.8f}')
        return root
    
    # b. brentq
    it = 0
    for it in range(max_iter):

        if fa * fb < 0:
            
            blk = a
            fblk = fa
            sa = sb = b - a

        if abs(fblk) < abs(fb):

            a = b
            b = blk
            blk = a

            fa = fb
            fb = fblk
            fblk = fa

        delta = (xtol + rtol * abs(b)) / 2
        sbis = (blk - b) / 2

        # root found
        if fb == 0 or abs(sbis) < delta:
            converged = True
            root = b
            break

        if abs(sa) > delta and abs(fb) < abs(fa):
            if a == blk:
                # interpolate
                stry = -fb * (b - a) / (fb - fa)
            else:
                # extrapolate
                da = (fa - fb) / (a - b)
                dblk = (fblk - fb) / (blk - b)
                stry = -fb * (fblk * dblk - fa * da) / \
                    (dblk * da * (fblk - fa))

            if (2 * abs(stry) < min(abs(sa), 3 * abs(sbis) - delta)):
                # good short step
                sa = sb
                sb = stry
            else:
                # bisect
                sa = sbis
                sb = sbis
        else:
            # bisect
            sa = sbis
            sb = sbis

        a = b
        fa = fb
        if (abs(sb) > delta):
            b += sb
        else:
            b += (delta if sbis > 0 else -delta)

        fb = f(b, *args)

        if do_print: print(f'{it:3d}: ',end='')
        if do_print: print(f'{varname} = {b:12.8f} -> {funcname} = {fb:12.8f}')

    if not converged: raise ValueError('brentq did not find solution')
    
    if do_print: print(f'\n{varname} = {root:12.8f} [{funcname} = {fb:12.8f}]\n')

    # final evaluation
    fb = f(root,*args)

    return root,fb
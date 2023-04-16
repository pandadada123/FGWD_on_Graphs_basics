# -*- coding: utf-8 -*-
"""
Optimization algorithms for OT
"""


import numpy as np
# from scipy.optimize.linesearch import scalar_search_armijo
from ot.lp import emd


class StopError(Exception):
    pass


class NonConvergenceError(Exception):
    pass
class StopError(Exception):
    pass
        

def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """Minimize over alpha, the function ``phi(alpha)``.
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57
    alpha > 0 is assumed to be a descent direction.
    Returns
    -------
    alpha
    phi1
    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0: # This is the formula for amijo backtracking
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1

def solve_1d_linesearch_quad_funct(a,b,c):
    # solve min f(x)=a*x**2+b*x+c sur 0,1
    f0=c
    df0=b
    f1=a+f0+df0

    if a>0: # convex
        minimum=min(1,max(0,-b/(2*a)))
        #print('entrelesdeux')
        return minimum
    else: # non convexe donc sur les coins
        if f0>f1:
            #print('sur1 f(1)={}'.format(f(1)))            
            return 1
        else:
            #print('sur0 f(0)={}'.format(f(0)))
            return 0

def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=0.99):
    """
    Armijo linesearch function that works with matrices
    find an approximate minimum of f(xk+alpha*pk) that satifies the
    armijo conditions.
    Parameters
    ----------
    f : function
        loss function
    xk : np.ndarray
        initial position
    pk : np.ndarray
        descent direction
    gfk : np.ndarray
        gradient of f at xk
    old_fval : float
        loss value at xk
    args : tuple, optional
        arguments given to f
    c1 : float, optional
        c1 const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float
        loss value at step alpha
    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval

    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0)

    return alpha, fc[0], phi1

def do_linesearch(cost,G,deltaG,Mi,f_val,amijo=True,C1=None,C2=None,reg=None,Gc=None,constC=None,M=None):
    #Gc= st
    #G=xt
    #deltaG=st-xt
    #Gc+alpha*deltaG=st+alpha(st-xt)
    """
    Solve the linesearch in the FW iterations
    Parameters
    ----------
    cost : method
        The FGW cost
    G : ndarray, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : ndarray (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : ndarray (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val :  float
        Value of the cost at G
    amijo : bool, optionnal
            If True the steps of the line-search is found via an amijo research. Else closed form is used.
            If there is convergence issues use False.
    C1 : ndarray (ns,ns), optionnal
        Structure matrix in the source domain. Only used when amijo=False
    C2 : ndarray (nt,nt), optionnal
        Structure matrix in the target domain. Only used when amijo=False
    reg : float, optionnal
          Regularization parameter. Corresponds to the alpha parameter of FGW. Only used when amijo=False
    Gc : ndarray (ns,nt)
        Optimal map found by linearization in the FW algorithm. Only used when amijo=False
    constC : ndarray (ns,nt)
             Constant for the gromov cost. See [3]. Only used when amijo=False
    M : ndarray (ns,nt), optionnal
        Cost matrix between the features. Only used when amijo=False
    Returns
    -------
    alpha : float
            The optimal step size of the FW
    fc : useless here
    f_val :  float
             The value of the cost for the next iteration
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if amijo:
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, Mi, f_val)
    else:
        dot1=np.dot(C1,deltaG) 
        dot12=dot1.dot(C2) # C1 dt C2
        a=-2*reg*np.sum(dot12*deltaG) #-2*alpha*<C1 dt C2,dt> si qqlun est pas bon c'est lui
        b=np.sum((M+reg*constC)*deltaG)-2*reg*(np.sum(dot12*G)+np.sum(np.dot(C1,G).dot(C2)*deltaG)) 
        c=cost(G) #f(xt)

        alpha=solve_1d_linesearch_quad_funct(a,b,c)
        fc=None
        f_val=cost(G+alpha*deltaG)
        
    return alpha,fc,f_val

def cg(a, b, M, reg, f, df, G0=None, numitmax=500, stopThr=1e-09, verbose=False,log=False,amijo=True,C1=None,C2=None,constC=None):
    """
    Solve the general regularized OT problem with conditional gradient
        The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg*f(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    G0 :  np.ndarray (ns,nt), optional
        initial guess (default is indep joint density)
    numitmax : int, optional
        Max number of itations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along itations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """

    loop = 1

    if log:
        log = {'loss': [],'delta_fval': [],'gradient':[], 'descent':[],'G':[],'alpha':[], 'Gc':[]}

    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0

    def cost(G):
        return np.sum(M * G) + reg * f(G) # for GWD, M is zero matrix. and reg=1

    f_val = cost(G) #f(xt)

    if log:
        log['loss'].append(f_val)
        log['G'].append(G)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0))

    while loop:

        it += 1
        old_fval = f_val
        #G=xt
        # problem linearization
        temp = df(G)
        Mi = M + reg * df(G) #Gradient(xt)
        # set M positive
        Mi += Mi.min() 
        # Mi-= Mi.min() 
        # Mi += np.min(Mi) # is the same as original one

        # solve linear program (Wasserstein is LP)
        Gc = emd(a, b, Mi) #st  actually same as EMD problem, minimizeg the inner product to find the argmin(Gc), has to satisfy the constraints of a and b
                            # 只是利用EMD的结构特点计算 Also within the constraints
        deltaG = Gc - G # dt:deltaG : ndarray (ns,nt) Difference between the optimal map found by linearization in the FW algorithm and the value at a given itation  
                        # Gc is the argmin matrix and G is the old one
                        # This is the descent direction = pk
                        # deltaG: descent direction; Mi: gradient
                        
        # argmin_alpha f(xt+alpha dt)  # find the step size alpha # Frank-Wolfe with linesearch?
        alpha, fc, f_val = do_linesearch(cost=cost,G=G,deltaG=deltaG,Mi=Mi,f_val=f_val,amijo=amijo,constC=constC,C1=C1,C2=C2,reg=reg,Gc=Gc,M=M)

        
        # alpha = 2/(it+2)
        
        if alpha is None or np.isnan(alpha) :
            raise NonConvergenceError('Alpha was not found')
        else:
            G = G + alpha * deltaG #xt+1=xt +alpha dt

        # test convergence
        if it >= numitmax:
            loop = 0
            
        delta_fval = (f_val - old_fval)

        #delta_fval = (f_val - old_fval)/ abs(f_val)
        #print(delta_fval)
        if abs(delta_fval) < stopThr:    # check if can stop
            loop = 0

        if log:  # add new results to log
            log['loss'].append(f_val)
            log['delta_fval'].append(delta_fval)
            log['gradient'].append(Mi)
            log['descent'].append(deltaG)
            log['G'].append(G)
            log['alpha'].append(alpha)
            log['Gc'].append(Gc)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}'.format(
                    'it.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
            print('{:5d}|{:8e}|{:8e}|{:5e}'.format(it, f_val, delta_fval,alpha))

    if log:
        return G, log
    else:
        return G
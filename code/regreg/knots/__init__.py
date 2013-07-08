"""
This module contains functions to solve two problems:

.. math::

        \minimize_{z,y} y^Ta

    subject to $(z,y) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$. 

.. math::

        \minimize_{z,y,w} y^Ta

    subject to $(z,w) \in \mathbf{epi}({\cal P})$ and the equality constraints
    $z-b^Ty=s, w=Dy$.

"""

import warnings
import numpy as np
import regreg.api as rr, regreg.affine as ra
from .admm import admm

def epigraph_linear(epigraph, D, Q, initial_primal=None, initial_dual=None,
                    epsilon=[1.e-5]*100, min_iters=20):
    D = rr.astransform(D)
    L = ra.product([D, rr.identity(1)])
    P = rr.linear_atom(epigraph, L)
    Z = rr.zero(L.input_shape)
    Z.quadratic = Q
    return rr.nesta(None, Z, P, initial_primal=initial_primal, initial_dual=initial_dual)

def linear_fractional_tfocs(a, b, epigraph, sign=1., tol=1.e-5, max_its=1000, epsilon=[1.e-4]*100, min_iters=10):
    """
    Solve the problem

    .. math::

        \minimize_{y,z} y^Ta

    subject to $(y,z) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$ with s in [1,-1].

    Inputs
    ======

    a, b: np.float

    sign: np.float (usually [1,-1])

    epigraph : 
          epigraph constraint whose proximal map projects onto the epigraph

    epsilon : np.float
          a sequence of epsilons for the smoothing

    tol : float
          When to stop, based on objective value decreasing.

    min_iters : int
          minimum number of NESTA iterations

    Outputs
    =======

    y : np.float

    z : float
         Solution with y = yz[:-1], z = yz[-1], and x can be found as x = y / z

    value : float
         The optimal value
    """

    a, b = np.asarray(a), np.asarray(b)
    p = a.shape[0]
    w_0, w_prev = np.zeros((2,) + epigraph.shape)
    
    value = np.inf
    for idx, eps in enumerate(epsilon):
        w_next, updated = solve_dual_block(b, a, w_0, eps, epigraph, tol=tol, max_iters=max_iters, sign=sign)[2:]
        if np.fabs(value - updated) < tol * np.fabs(updated) and idx >= min_iters:
            break
        w_0 = update_w_0(w_next, w_prev, idx)
        w_prev, value = w_next, updated
    return value, None

def linear_fractional_nesta(a, b, epigraph, sign=1., tol=1.e-5, 
                            max_iters=1000, epsilon=[1.e-4]*10, min_iters=10,
                            initial_primal=None,
                            initial_dual=None):
    """
    Solve the problem

    .. math::

        \minimize_{y,z} y^Ta

    subject to $(y,z) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$ with s in [1,-1].

    Inputs
    ======

    a, b: np.float

    sign: np.float (usually [1,-1])

    epigraph : 
          epigraph constraint whose proximal map projects onto the epigraph

    epsilon : np.float
          a sequence of epsilons for the smoothing

    tol : float
          When to stop, based on objective value decreasing.

    min_iters : int
          minimum number of NESTA iterations

    Outputs
    =======

    y : np.float

    z : float
         Solution with y = yz[:-1], z = yz[-1], and x can be found as x = y / z

    value : float
         The optimal value
    """

    a, b = np.asarray(a), np.asarray(b)
    p = a.shape[0]
    a_full = np.zeros(p+1)
    a_full[:-1] = a
    
    b_full = np.zeros(p+1)
    b_full[:-1] = -b
    b_full[-1] = 1.

    linear_constraint = rr.zero_constraint.affine(b_full.reshape((1,-1)), -sign)

    Q = rr.identity_quadratic(0,0,a_full,0)
    oldQ, epigraph.quadratic = epigraph.quadratic, Q
    coef = rr.nesta(None, epigraph, linear_constraint, epsilon=epsilon, tol=tol,
                    min_iters=min_iters,
                    coef_tol=tol,
                    initial_primal=initial_primal,
                    initial_dual=initial_dual)[0]
    epigraph.quadratic = oldQ
    return (coef[:-1]*a).sum(), coef[:-1] / coef[-1]

def linear_fractional_admm(a, b, epigraph, sign=1., tol=1.e-5, rho=1, 
                           max_iters=1000, min_iters=10,
                           initial=None):
    """
    Solve the problem

    .. math::

        \minimize_{y,z} y^Ta

    subject to $(y,z) \in \mathbf{epi}({\cal P})$ and the equality constraint 
    $z-b^Ty=s$ with s in [1,-1].

    Inputs
    ======

    a, b: np.float

    sign: np.float (usually [1,-1])

    epigraph : 
          epigraph constraint whose proximal map projects onto the epigraph

    epsilon : np.float
          a sequence of epsilons for the smoothing

    tol : float
          When to stop, based on objective value decreasing.

    rho : float
          ADMM parameter

    min_iters : int
          minimum number of ADMM iterations

    max_iters : int
          minimum number of ADMM iterations

    Outputs
    =======

    y : np.float

    z : float
         Solution with y = yz[:-1], z = yz[-1], and x can be found as x = y / z

    value : float
         The optimal value
    """

    a, b = np.asarray(a), np.asarray(b)
    p = a.shape[0]
    a_full = np.zeros(p+1)
    a_full[:-1] = a
    
    b_full = np.zeros(p+1)
    b_full[:-1] = -b
    b_full[-1] = 1.

    n_full = b_full / np.linalg.norm(b_full)

    linear_constraint = rr.projection_complement((p+1,),
                                                 n_full.reshape((1,-1)),
                                                 offset=sign * b_full / np.linalg.norm(b_full)**2) 

    if initial is not None:
        epigraph.coefs[:] = initial
        linear_constraint.coefs[:] = initial

    Q = rr.identity_quadratic(0,0,a_full,0)
    linear_constraint.quadratic = Q

    soln = admm(linear_constraint, epigraph, tol=tol, min_iters=min_iters,
                max_iters=max_iters,
                rho=rho)
    value = linear_constraint.objective(soln) + epigraph.objective(soln)

    return value, soln[:-1] / soln[-1]

def find_alpha(soln, X, tangent_vectors=None):
    """
    Return $P_X(\eta), \alpha_{\eta}$
    """
    X = rr.astransform(X)
    
    if tangent_vectors is not None:
        tangent_vectors = np.asarray(tangent_vectors)
        tangent_vectors = np.array([X.linear_map(tv).copy() for tv in tangent_vectors])
        W = np.array(tangent_vectors)
        Winv = np.linalg.pinv(W)
        P = np.dot(W.T, Winv.T)
    else:
        P = 0
    
    alpha = X.linear_map(soln).copy()
    alpha = alpha - np.dot(P, alpha)
    conditional_variance = (alpha**2).sum() 

    alpha /= (alpha**2).sum()
    alpha = X.adjoint_map(alpha).copy()

    return alpha, conditional_variance

def eta_step(gh, b, a, w_0, mu, sign=1):
    y_0, z_0 = w_0[:-1], w_0[-1]
    g, h = gh[:-1], gh[-1]
    
    num = mu*z_0 - h + ((g+a-mu*y_0)*b).sum() - sign * mu
    den = 1 + (b**2).sum()
    return num / den

def gh_step(eta, b, a, w_0, mu, epigraph):
    y_0, z_0 = w_0[:-1], w_0[-1]
    W = np.zeros_like(w_0)
    W[:-1] = mu*y_0 + eta*b - a
    W[-1] = mu*z_0 - eta
    U = epigraph.cone_prox(W)
    return W - U

def solve_dual_block(b, a, w_0, mu, epigraph, sign=1, tol=1.e-6, max_iters=1000):
    eta = 0
    gh = np.zeros_like(w_0)
    
    for idx in range(max_iters):
        new_eta = eta_step(gh, b, a, w_0, mu, sign=sign)
        new_gh = gh_step(new_eta, b, a, w_0, mu, epigraph)
        if ((np.linalg.norm(new_gh - gh) < tol * max(1, np.linalg.norm(gh)))
            and (np.fabs(new_eta - eta) < tol * max(1, np.fabs(eta)))):
            break
        eta = new_eta
        gh = new_gh
        
    if idx == max_iters-1:
        warnings.warn('algorithm did not converge after %d steps' % max_iters)
    
    g, h = gh[:-1], gh[-1]
    primal = np.zeros_like(w_0)
    primal += w_0
    primal[:-1] -= (g -eta*b + a) / mu
    primal[-1] -= (h + eta) / mu
    
    return gh, eta, primal, (a*primal[:-1]).sum()

def update_w_0(w_star, w_star_last, j):
    return w_star + j / (j+3.) * (w_star - w_star_last)


import numpy as np, os
from scipy.stats import chi

import regreg.api as rr
from regreg.knots import (find_C_X, linear_fractional_admm,
                          linear_fractional_tfocs,
                          linear_fractional_admm,
                          chi_pvalue)
from lasso import signed_basis_vector

def glasso_knot(X, R, groups, soln, 
                epsilon=([1.e-2] + [1.e-4]*3 + [1.e-5]*3 + 
                         [1.e-6]*50 + [1.e-8]*200)
                , tol=1.e-7, method='admm', weights=None,
                min_iters=10):
    """
    Find an approximate LASSO knot
    """

    dual = rr.group_lasso_dual(groups, weights=weights, lagrange=1.)
    primal = rr.group_lasso(groups, weights=weights, lagrange=1.)

    X = rr.astransform(X)
    U = X.adjoint_map(R).copy()
    terms = dual.terms(U)
    imax = np.argmax(terms)
    L = terms[imax]
    gmax = dual.group_labels[imax]
    wmax = dual.weights[gmax]

    # for this below we are assuming uniqueness of first group
    which = dual.groups == gmax
    kmax = which.sum()

    soln = np.zeros(X.input_shape)
    soln[which] = (U[which] / np.linalg.norm(U[which])) / wmax

    p = primal.shape[0]
    if kmax > 1:
        tangent_vectors = [signed_basis_vector(v, 1, p) for v in np.nonzero(which)[0]][:-1]
        for tv in tangent_vectors:
            tv[:] = tv - np.dot(tv, soln) * soln / np.linalg.norm(soln)**2
    else:
        tangent_vectors = None
    C_X, var = find_C_X(soln, X, tangent_vectors)
    
    # in actually finding M^+ we don't have to subtract the conditional 
    # expectation of the tangent part, as it will be zero at eta that
    # achieves L1

    p = soln.shape[0]
    epigraph = rr.group_lasso_epigraph(groups, weights=weights)

    if method == 'nesta':
        Mplus, next_soln = linear_fractional_nesta(-(U-C_X*L), 
                                                    C_X, 
                                                    epigraph, 
                                                    tol=tol,
                                                    epsilon=epsilon,
                                                    initial_primal=initial_primal,
                                                    min_iters=min_iters)
    elif method == 'tfocs':
        Mplus, next_soln = linear_fractional_tfocs(-(U-C_X*L), 
                                                    C_X, 
                                                    epigraph, 
                                                    tol=tol,
                                                    epsilon=epsilon,
                                                    min_iters=min_iters)
    elif method == 'admm':
        Mplus, next_soln = linear_fractional_admm(-(U-C_X*L), 
                                                   C_X, 
                                                   epigraph, 
                                                   tol=tol,
                                                   rho=np.sqrt(p),
                                                   min_iters=min_iters)
    elif method == 'explicit':
        a = U - C_X * L
        b = C_X

        Vplus = []
        Vminus = []
        for label in dual.group_labels:
            if label != gmax:
                group = dual.groups == label
                weight = dual.weights[label]
                tf = trignometric_form(a[group], b[group], weight)
                Vplus.append(tf[0])
                Vminus.append(tf[1])
        Mplus, next_soln, Mminus = -np.nanmax(Vplus), None, np.nanmin(Vminus)

    else:
        raise ValueError('method must be one of ["nesta", "tfocs", "admm", "explicit"]')

    if dual.seminorm(C_X) > 1.001:
        if method == 'nesta':
            Mminus, _ = linear_fractional_nesta(-(U-C_X*L), 
                                                 C_X, 
                                                 epigraph, 
                                                 tol=tol,
                                                 sign=-1,
                                                 epsilon=epsilon,
                                                 initial_primal=initial_primal,
                                                 min_iters=min_iters)
        elif method == 'tfocs':
            Mminus, _ = linear_fractional_tfocs(-(U-C_X*L), 
                                                 C_X, 
                                                 epigraph, 
                                                 tol=tol,
                                                 sign=-1,
                                                 epsilon=epsilon,
                                                 initial_primal=initial_primal,
                                                 min_iters=min_iters)
            
        elif method == 'admm':
            Mminus, _ = linear_fractional_admm(-(U-C_X*L), 
                                                C_X, 
                                                epigraph, 
                                                tol=tol,
                                                sign=-1,
                                                rho=np.sqrt(p),
                                                min_iters=min_iters)
        elif method == 'explicit':
            pass
        else:
            raise ValueError('method must be one of ["nesta", "tfocs", "admm", "explicit"]')
    else:
        Mminus = np.inf

    return (L, -Mplus, Mminus, C_X, tangent_vectors, var, U, C_X, next_soln, kmax, wmax)

def simulate_random(n, g, k, orthonormal=True, beta=None, max_size=None):
    """
    Generate a random group of design matrices
    of random sizes with optionally orthonormal columns.
    Optionally, add signal and cap the size of the groups at some size.
    
    Parameters
    ----------

    n: number of rows

    g: number of groups

    k: before capping, group sizes are IID Poisson with this parameter + 1

    orthonormal: make the columns orthonormal?

    beta: add a signal to response

    max_size: optional largest group size

    >>> np.random.seed(1)
    >>> X, Y, G, W = simulate_random(100, 6, 4)
    >>> X.shape
    (100, 27)
    >>> Y.shape
    (100,)
    >>> G
    [slice(0, 3, None), slice(3, 6, None), slice(6, 10, None), slice(10, 15, None), slice(15, 20, None), slice(20, 27, None)]
    >>> W
    array([ 1.55920972,  1.49272083,  1.93652742,  1.00009695,  1.00956689,
            1.25195142])
    >>> 


    """
    group_sizes = np.random.poisson(k, g) + 1
    if max_size is not None:
        group_sizes = np.clip(group_sizes, 0, max_size)
    p = np.sum(group_sizes)
    Gsum = np.cumsum(group_sizes)
    groups = [slice(0, Gsum[0])]
    for i in range(g-1):
        groups.append(slice(Gsum[i], Gsum[i+1]))

    X = np.random.standard_normal((n,p))
    A = np.arange(p) * (2* np.random.binomial(1,0.5, size=(p,)) - 1)
    np.random.shuffle(A)
    X = X * A[np.newaxis,:]
    X -= X.mean(0)
    if beta is None:
        beta = np.zeros(p)

    if orthonormal:
        for group in groups:
            X[:,group] = np.linalg.svd(X[:,group], full_matrices=False)[0]
    else:
        for group in groups:
            X[:,group] /= np.linalg.norm(X[:,group])

    Y = np.dot(X, beta) + np.random.standard_normal(n)
    weights = np.random.sample(g) + 1
    G = np.zeros(p)
    W = {}
    for i, g in enumerate(groups):
        G[g] = i
        W[i] = weights[i]
    return X, Y, G, W
    
def simulate_fixed(n, g, k, orthonormal=True, beta=None, max_size=None, useA=True):
    """
    Generate a random group of design matrices of fixed size with orthonormal columns.
    Optionally, add signal and cap the size of the groups at some size.
    
    Parameters
    ----------

    n: number of rows

    g: number of groups

    k: group_size

    orthonormal: make the columns orthonormal?

    beta: add a signal to response

    max_size: optional largest group size

    useA : bool
        If True, tries to add some structure to the design matrix.

    Returns
    -------

    X : np.ndarray(np.float)
        Design matrix.

    Y : np.ndarray(np.float)
        Response vector.

    groups : [[int]]
        Sequence of groups of variables.

    weights : [float]
        Sequence of weights for each group.

    >>> X, Y, G, W = simulate_fixed(100, 3,4)
    >>> X.shape
    (100, 12)
    >>> Y.shape
    (100,)
    >>> G
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    >>> W
    array([ 2.,  2.,  2.])
    >>> 

    """

    p = g*k
    X = np.random.standard_normal((n,p))
    A = np.arange(p) * (2* np.random.binomial(1,0.5, size=(p,)) - 1)
    np.random.shuffle(A)
    
    groups = np.arange(g*k).reshape((g,k))
    groups = [list(group) for group in groups]
    
    if useA:
        X = X * A[np.newaxis,:]

    if beta is None:
        beta = np.zeros(p)
    
    if orthonormal:
        for group in groups:
            X[:,group] = np.linalg.svd(X[:,group], full_matrices=False)[0]
    X -= X.mean(0)
    Y = np.dot(X, beta) + np.random.standard_normal(n)
    
    weights = np.ones(g) 
    weights = np.sqrt(k) * np.ones(g)
    return X, Y, groups, weights

def trignometric_form(num, den, weight, tol=1.e-6):
    a, b, w = num, den, weight # shorthand

    if np.linalg.norm(a) / np.linalg.norm(b) < tol:
        return 0, np.inf

    Ctheta = np.clip((a*b).sum() / np.sqrt((a**2).sum() * (b**2).sum()), -1, 1)
    Stheta = np.sqrt(1-Ctheta**2)
    theta = np.arccos(Ctheta)

    Sphi = np.linalg.norm(b) * Stheta / w
    phi1 = np.arcsin(Sphi)
    phi2 = np.pi - phi1

    V1 = np.linalg.norm(a) * np.cos(phi1) / (w - np.linalg.norm(b) * np.cos(theta-phi1))
    V2 = np.linalg.norm(a) * np.cos(phi2) / (w - np.linalg.norm(b) * np.cos(theta-phi2))

    if np.isnan(V1) or np.isnan(V2):
        stop
    if np.linalg.norm(b) < w:
        return max([V1,V2]), np.inf
    else:
        return min([V1,V2]), max([V1,V2])

def exp_pvalue(L, Mplus, Mminus, sd):
    '''
    exponential approximation
    '''
    return np.exp(-L*(L-Mplus)/sd**2)


def first_test(X, Y, groups, weights={}, nsim=50000,
               method='MC',
               sigma=1):
    soln = np.zeros(X.shape[1])
    (L, Mplus, Mminus, _, _, 
     var, _, _, _, k, w) = glasso_knot(X, Y, groups, 
                                       soln,
                                       method='explicit',
                                       weights=weights)
    sd = np.sqrt(var) * sigma
    p = pvalue(L, Mplus, Mminus, sd, k, method=method, nsim=nsim)
    return p, exp_pvalue(L, Mplus, Mminus, sd)

def pvalue(L, Mplus, Mminus, sd, k, method='MC', nsim=1000):
    return chi_pvalue(L, Mplus, Mminus, sd, k, method=method, nsim=nsim)

def check_knots(nsim=50, seed=0, orthonormal=False, null=True):

    shape_nn = (100, 20)

    np.random.seed(seed)
    def solve_group_lasso(X, Y, groups, weights, L, tol=1.e-5, min_its=10):
        """
        Solve the nuclear norm problem with design matrix X, outcome Y
        and Lagrange parameter L.
        """
        X = rr.astransform(X)
        loss = rr.squared_error(X, Y)
        penalty = rr.group_lasso(groups, weights=weights, lagrange=L)
        problem = rr.simple_problem(loss, penalty)
        soln = problem.solve(tol=tol, min_its=min_its)
        resid = Y - X.linear_map(soln).copy()
        return soln, resid

    def find_next_knot_gl(X, R, soln, groups, weights, L, multiplicity, niter=50, verbose=False):

        loss = rr.squared_error(X, R)
        penalty = rr.group_lasso(groups, weights=weights, lagrange=L)
        dual = rr.group_lasso_dual(groups, weights=weights, lagrange=1)
        L2 = L
        for _ in range(niter):
            grad = loss.smooth_objective(soln, mode='grad')
            Lcandidate = (sorted(dual.terms(grad))[-2]  +  L2) / 2.
            soln = solve_group_lasso(X, R, groups, weights, Lcandidate, tol=1.e-13,
                                     min_its=1000)[0]
            L2 = Lcandidate
            grad = loss.smooth_objective(soln, mode='grad')
            print sorted(dual.terms(grad))[-3:], Lcandidate
        return L, L2

    if not os.path.exists('group_lasso_knots.npy'):

        n, g, k = 100, 10, 5
        values = []
        for i in range(nsim):
            X, Y, groups, weights = simulate_random(n, g, k, orthonormal=orthonormal)
            if not null:
                beta = np.zeros(X.shape[1])
                beta[groups == 0] = np.random.standard_normal((groups == 0).sum()) * 10
                Y += np.dot(X, beta)
            soln = np.zeros(X.shape[1])
            dual = rr.group_lasso_dual(groups, weights=weights, lagrange=1)
            penalty = rr.group_lasso(groups, weights=weights, lagrange=1)
            L = dual.seminorm(np.dot(X.T, Y))
            loss = rr.squared_error(X, Y)
            strong_rules_knot = find_next_knot_gl(X, Y, soln, groups, weights, L, 1, verbose=False,
                                                  niter=50)[1]
            Mplus = glasso_knot(X, Y, groups, soln,
                                weights=weights,
                                method='admm',
                                min_iters=300,
                                tol=1.e-12)[1]

            Mplus2 = glasso_knot(X, Y, groups, soln,
                                 weights=weights,
                                 method='explicit')[1]

            soln = solve_group_lasso(X, Y, groups, weights, Mplus2, tol=1.e-13, 
                                     min_its=1000)[0]
            grad = loss.smooth_objective(soln, mode='grad')
            print sorted(penalty.terms(soln))[-3:], 'terms'
            print L, Mplus
            values.append([Mplus, Mplus2, strong_rules_knot])
            print values[-1]

        values = np.array(values)
        np.save('group_lasso_knots.npy', np.array(values))
    else:
        values = np.load('group_lasso_knots.npy')

    from matplotlib import pyplot as plt

    plt.clf()
    plt.scatter(values[:,0], values[:,1], label=r'ADMM vs. explicit')
    plt.legend(loc='lower right')
    plt.savefig('group_lasso_knots1.pdf')

    plt.clf()
    plt.scatter(values[:,0], values[:,2], c='r', label=r'ADMM vs. $\lambda_2$')
    plt.legend(loc='lower right')
    plt.savefig('group_lasso_knots2.pdf')


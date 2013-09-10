import numpy as np

import regreg.api as rr
from regreg.knots import (find_C_X, 
                          linear_fractional_nesta, 
                          linear_fractional_tfocs,
                          linear_fractional_admm,
                          chi_pvalue)

def lasso_knot_covstat(X, R, soln, tol=1.e-6):
    """
    Find an approximate LASSO knot
    """
    X = rr.astransform(X)
    p = X.input_shape[0]
    l1soln = np.fabs(soln).sum()
    if l1soln > 0:
        soln = soln / l1soln
    U = X.adjoint_map(R).copy()
    L = np.fabs(U).max()
    which = np.nonzero(np.fabs(np.fabs(U) - L) < tol * L)[0]
    if l1soln == 0:
        soln = np.zeros(X.input_shape)
        soln[which] = np.sign(U)[which] / which.shape[0]
    s = np.sign(soln)

    if which.shape[0] > 1:
        tangent_vectors = [signed_basis_vector(v, s[v], p) - soln for v in which[1:]]
    else:
        tangent_vectors = None

    C_X, var = find_C_X(soln, X, tangent_vectors)
    
    L = np.fabs(U).max()

    # having found C_X, we can 
    # solve for M+, M- explicitly (almost surely)
    Mplus = {}
    Mminus = {}
    keep = np.ones(U.shape[0], np.bool)
    keep[which] = 0

    den = 1 - C_X
    num = U - C_X * L
    Mplus[1] = (num / den * (den > 0))[keep]
    Mminus[1] = (num * keep / (den + (1 - keep)))[den < 0]
    
    den = 1 + C_X
    num =  -(U - C_X * L)
    Mplus[-1] = (num / den * (den > 0))[keep]
    Mminus[-1] = (num * keep / (den + (1 - keep)))[den < 0]
    
    mplus = np.hstack([Mplus[1],Mplus[-1]])
    Mplus = np.max(mplus[mplus < L]) # this check may not be necessary
    
    mminus = []
    if Mminus[1].shape:
        mminus.extend(list(Mminus[1]))
    if Mminus[-1].shape:
        mminus.extend(list(Mminus[-1]))
    if mminus:
        mminus = np.array(mminus)
        mminus = mminus[mminus > L]
        if mminus.shape != (0,):
            Mminus = mminus.min()
        else:
            Mminus = np.inf
    else:
        Mminus = np.inf
        
    return (L, Mplus, Mminus, C_X, tangent_vectors, var, U, C_X, None)

def solve_lasso(X, Y, L, tol=1.e-5):
    """
    Solve the nuclear norm problem with design matrix X, outcome Y
    and Lagrange parameter L.
    """
    n, p = X.shape
    X = rr.astransform(X)
    loss = rr.squared_error(X, Y)
    penalty = rr.l1norm(p, lagrange=L)
    problem = rr.simple_problem(loss, penalty)
    soln = problem.solve(tol=tol)
    resid = Y - X.linear_map(soln).copy()
    return soln, resid

def find_next_knot_lasso(X, R, soln, L, niter=20, verbose=False):
    
    loss = rr.squared_error(X, R)
    grad = loss.smooth_objective(soln, mode='grad')

    L2 = L
    for _ in range(niter):
        grad = loss.smooth_objective(soln, mode='grad')
        Lcandidate = (sorted(np.fabs(grad))[-2] + L2) / 2.
        soln = solve_lasso(X, R, Lcandidate)[0]
        L2 = Lcandidate
    
    return L, L2


def signed_basis_vector(j, sign, p):
    v = np.zeros(p)
    v[j] = sign
    return v

def lasso_knot(X, R, soln, 
               epsilon=([1.e-2] + [1.e-4]*3 + [1.e-5]*3 + 
                        [1.e-6]*50 + [1.e-8]*200), tol=1.e-7, 
               method='admm',
               min_iters=10):
    """
    Find an approximate LASSO knot
    """

    X = rr.astransform(X)
    p = X.input_shape[0]
    l1soln = np.fabs(soln).sum()
    if l1soln > 0:
        soln = soln / l1soln
    U = X.adjoint_map(R).copy()
    L = np.fabs(U).max()
    which = np.nonzero(np.fabs(np.fabs(U) - L) < tol * L)[0]
    if l1soln == 0:
        soln = np.zeros(X.input_shape)
        soln[which] = np.sign(U)[which] / which.shape[0]
    s = np.sign(soln)

    if which.shape[0] > 1:
        tangent_vectors = [signed_basis_vector(v, s[v], p) - soln for v in which[1:]]
    else:
        tangent_vectors = None

    C_X, var = find_C_X(soln, X, tangent_vectors)
    
    # in actually finding M^+ we don't have to subtract the conditional 
    # expectation of the tangent part, as it will be zero at eta that
    # achieves L1

    p = soln.shape[0]
    epigraph = rr.l1_epigraph(p+1)

    initial_primal = np.zeros(p+1)
    initial_primal[:-1] = soln
    initial_primal[-1] = np.fabs(soln).sum()

    U = X.adjoint_map(R).copy()
    L = np.fabs(U).max()
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
        Mplus, Mminus = lasso_knot_covstat(X, R, soln)[1:3]
        Mplus = -Mplus
        next_soln = None
    else:
        raise ValueError('method must be one of ["nesta", "tfocs", "admm", "explicit"]')

    if np.fabs(C_X).max() > 1.001:
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

    return (L, -Mplus, Mminus, C_X, tangent_vectors, var, U, C_X, next_soln)

def test_main():

    import rpy2.robjects as rpy

    try:
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
    except ImportError:
        pandas2ri = None
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

    n, p = (100, 200)

    X_lasso = np.random.standard_normal((n,p))

    beta_lasso = np.zeros(p)
    beta_lasso[:3] = [0.1,1.5,2.]
    Y_lasso = np.random.standard_normal(n) + np.dot(X_lasso, beta_lasso)

    lagrange_lasso = 0.995 * np.fabs(np.dot(X_lasso.T,Y_lasso)).max()
    soln_lasso, resid_lasso = solve_lasso(X_lasso, Y_lasso, lagrange_lasso, tol=1.e-10)

    find_next_knot_lasso(X_lasso, resid_lasso, soln_lasso, lagrange_lasso)

    L, Mplus, Mminus, C_X, tv, var, U, C_X, _ = lasso_knot(X_lasso, resid_lasso, soln_lasso)
    print Mplus, Mminus
    print ((L, Mplus), find_next_knot_lasso(X_lasso, resid_lasso, soln_lasso, lagrange_lasso))
    print lasso_knot_covstat(X_lasso, resid_lasso, soln_lasso)[:3]

    rpy.r.assign('X', X_lasso)
    rpy.r.assign('Y', Y_lasso)
    L = rpy.r("""
    Y = as.numeric(Y)
    library(lars)
    L = lars(X, Y, type='lasso', intercept=FALSE, normalize=FALSE, max.steps=10)
    L$lambda
    """)

    print L[:2]

    def trendD(m, order=1):
        if order == 1:
            return -(np.diag(np.ones(m)) - np.diag(np.ones(m-1),1))[:-1]
        else:
            d = np.identity(m)
            for j in range(order):
                c = trendD(m-j,1)
                d = np.dot(c,d)
            return d

    m = 1000
    X_lasso = np.linalg.pinv(trendD(m, order=2))
    m, p = X_lasso.shape
    Y_lasso = np.random.standard_normal(m)

    lagrange_lasso = 0.99 * np.fabs(np.dot(X_lasso.T, Y_lasso)).max()
    soln_lasso, resid_lasso = solve_lasso(X_lasso, Y_lasso, lagrange_lasso, tol=1.e-13)
    L, Mplus, Mminus, C_X, tv, var = lasso_knot(X_lasso, resid_lasso, soln_lasso)
    print Mplus, Mminus
    print lasso_knot_covstat(X_lasso, resid_lasso, soln_lasso)[:3]

    # print ((L, Mplus), find_next_knot_lasso(X_lasso, resid_lasso, soln_lasso, lagrange_lasso))
    # rpy.r.assign('X', X_lasso)
    # rpy.r.assign('Y', Y_lasso)
    # L = rpy.r("""
    # Y = as.numeric(Y)
    # library(lars)
    # L = lars(X, Y, type='lasso', intercept=FALSE, normalize=FALSE)
    # L$lambda
    # """)

    # print L[:5]

    # soln_lasso2 = solve_lasso(X_lasso, Y_lasso, 0.97*L[1], tol=1.e-13)[0]
    # print np.nonzero(soln_lasso2)[0]

    # soln_lasso3 = solve_lasso(X_lasso, Y_lasso, 0.97*Mplus, tol=1.e-13)[0]
    # print np.nonzero(soln_lasso3)[0]

    # soln_lasso4 = solve_lasso(X_lasso, Y_lasso, 1.03*Mplus, tol=1.e-13)[0]
    # print np.nonzero(soln_lasso4)[0]

if __name__ == '__main__':
    test_main()

def pvalue(L, Mplus, Mminus, sd, method='cdf', nsim=1000):
    return chi_pvalue(L, Mplus, Mminus, sd, 1, method=method, nsim=nsim)

def exp_pvalue(L, Mplus, Mminus, sd):
    '''
    exponential approximation
    '''
    return np.exp(-L*(L-Mplus)/sd**2)

def first_test(X, Y, nsim=50000,
               method='cdf',
               sigma=1):
    soln = np.zeros(X.shape[1])
    (L, Mplus, Mminus, _, _, 
     var, _, _, _) = lasso_knot(X, Y, 
                                soln,
                                method='explicit')
    sd = np.sqrt(var) * sigma
    p = pvalue(L, Mplus, Mminus, sd, method=method, nsim=nsim)
    return p, exp_pvalue(L, Mplus, Mminus, sd)
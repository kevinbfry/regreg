import regreg.api as rr
import numpy as np

from regreg.knots import find_alpha, linear_fractional_nesta
from .lasso import find_next_knot_lasso, lasso_knot_covstat

def solve_glasso(X, Y, L, tol=1.e-5):
    """
    Solve the nuclear norm problem with design matrix X, outcome Y
    and Lagrange parameter L.
    """
    n, p = X.shape
    X = rr.astransform(X)
    loss = rr.squared_error(X, Y)
    penalty = rr.group_lasso(np.arange(p), lagrange=L)
    problem = rr.simple_problem(loss, penalty)
    soln = problem.solve(tol=tol)
    resid = Y - X.linear_map(soln).copy()
    return soln, resid

def glasso_knot(X, R, soln, L, epsilon=[1.e-2] + [1.e-4]*3 + [1.e-5]*3 + [1.e-6]*50 + [1.e-8]*200):
    """
    Find an approximate LASSO knot
    """
    X = rr.astransform(X)
    p = X.input_shape[0]
    soln = soln / np.fabs(soln).sum()
    which = np.nonzero(soln)[0]
    s = np.sign(soln)

    if which.shape[0] > 1:
        tangent_vectors = [signed_basis_vector(v, s[v], p) - soln for v in which[1:]]
        print 'tan', len(tangent_vectors)
    else:
        tangent_vectors = None

    alpha, var = find_alpha(soln, X, tangent_vectors)
    
    # in actually finding M^+ we don't have to subtract the conditional 
    # expectation of the tangent parta, as it will be zero at eta that
    # achieves L1

    p = soln.shape[0]
    epigraph = rr.group_lasso_epigraph(np.arange(p))

    Mplus = linear_fractional_nesta(-(X.adjoint_map(R).copy()-alpha*L), 
                                        alpha, 
                                        epigraph, 
                                        tol=1.e-6,
                                        epsilon=epsilon,
                                        min_iters=10)

    if np.fabs(alpha).max() > 1.001:
        Mminus = linear_fractional_nesta(-(X.adjoint_map(R).copy()-alpha*L), 
                                             alpha, 
                                             epigraph, 
                                             tol=1.e-6,
                                             sign=-1,
                                             epsilon=epsilon,
                                             min_iters=10)

    else:
        Mminus = np.inf

    return (L, -Mplus, Mminus, alpha, tangent_vectors, var)

def test_main():

    import rpy2.robjects as rpy

    try:
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
    except ImportError:
        pandas2ri = None
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

    n, p = 100, 200
    X_lasso = np.random.standard_normal((n,p))

    beta_lasso = np.zeros(p)
    beta_lasso[:3] = [0.1,1.5,2.]
    Y_lasso = np.random.standard_normal(n) + np.dot(X_lasso, beta_lasso)

    penalty = rr.group_lasso(np.arange(p), lagrange=1)
    dual_penalty = penalty.conjugate

    lagrange_lasso = 0.995 * dual_penalty.seminorm(np.dot(X_lasso.T,Y_lasso), lagrange=1.)
    print lagrange_lasso, 0.995 * np.fabs(np.dot(X_lasso.T, Y_lasso)).max(), 'huh'


    soln_lasso, resid_lasso = solve_glasso(X_lasso, Y_lasso, lagrange_lasso, tol=1.e-10)


    L, Mplus, Mminus, alpha, tv, var = glasso_knot(X_lasso, resid_lasso, soln_lasso, lagrange_lasso)
    print Mplus, Mminus
    print ((L, Mplus), find_next_knot_lasso(X_lasso, resid_lasso, soln_lasso, lagrange_lasso))
    print lasso_knot_covstat(X_lasso, resid_lasso, soln_lasso, lagrange_lasso)[:3]

    rpy.r.assign('X', X_lasso)
    rpy.r.assign('Y', Y_lasso)
    L = rpy.r("""
    Y = as.numeric(Y)
    library(lars)
    L = lars(X, Y, type='lasso', intercept=FALSE, normalize=FALSE, max.steps=10)
    L$lambda
    """)

    print L[:2]

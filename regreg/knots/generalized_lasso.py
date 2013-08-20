import regreg.api as rr
import numpy as np

from regreg.knots import find_alpha, linear_fractional_nesta

def lasso_knot(X, R, D, soln, epsilon=[1.e-2] + [1.e-4]*3 + [1.e-5]*3 + [1.e-6]*50 + [1.e-8]*200, tol=1.e-7):
    """
    Find an approximate LASSO knot
    """
    X = rr.astransform(X)
    D = rr.astransform(D)

    n = X.output_shape[0]
    m = D.output_shape[0]
    p = X.input_shape[0]

    Dsoln = D.linear_map(soln)
    Dsoln = Dsoln / np.fabs(Dsoln).sum()

    which = np.nonzero(Dsoln)[0]
    s = np.sign(soln)

    if which.shape[0] > 1:
        tangent_vectors = [signed_basis_vector(v, s[v], p) - soln for v in which[1:]]
    else:
        tangent_vectors = None

    alpha, var = find_alpha(soln, X, tangent_vectors)
    
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
    Mplus = linear_fractional_nesta(-(U-alpha*L), 
                                        alpha, 
                                        epigraph, 
                                        tol=tol,
                                        epsilon=epsilon,
                                        initial_primal=initial_primal,
                                        min_iters=10)

    if np.fabs(alpha).max() > 1.001:
        Mminus = linear_fractional_nesta(-(U-alpha*L), 
                                             alpha, 
                                             epigraph, 
                                             tol=tol,
                                             sign=-1,
                                             epsilon=epsilon,
                                             initial_primal=initial_primal,
                                             min_iters=10)

    else:
        Mminus = np.inf

    return (L, -Mplus, Mminus, alpha, tangent_vectors, var)

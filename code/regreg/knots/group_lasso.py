import regreg.api as rr
import numpy as np

from regreg.knots import find_alpha, linear_fractional_nesta
from .lasso import find_next_knot_lasso, lasso_knot_covstat

def solve_glasso(X, Y, groups, L, tol=1.e-5):
    """
    Solve the nuclear norm problem with design matrix X, outcome Y
    and Lagrange parameter L.
    """
    n, p = X.shape
    X = rr.astransform(X)
    loss = rr.squared_error(X, Y)
    penalty = rr.group_lasso(groups, lagrange=L)
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

def maximum_pinned(X_h, w_h, X_g, w_g, P_g, y):
    """
    Compute the maximum within group $h$ knowing that group $g$ achieved
    $\lambda_1$.

    This function computes 

    .. math::

        \sup_{v_h:\|v_h\|_2=1} \frac{w_h^{-l}v_h^TX_h^T(I - P_g)y}{1 - \frac{w_g}{w_h} 
         \frac{(y^TX_gX_g^Ty)^{1/2}}{\|X_gX_g^Ty\|^2_2} v_h^TX_h^TX_g X_g^Ty}

    XXX TODO: return the maximizer as well -- should help for group LASSO paths.

    Parameters
    ----------

    X_h : np.ndarray(np.float)
        Design matrix for group $h$

    w_h : float
        Weight for group $h$

    X_g : np.ndarray(np.float)
        Design matrix for group $g$

    w_g : float
        Weight for group $g$

    P_g : np.ndarray(np.float)
        Projection onto column space of $X_g$. For now, 
        we have implicitly been assuming that $P_g=X_gX_g^T$.

    y : np.ndarray(np.float)
        The outcome.

    Returns
    -------

    value : float
        The achived supremum in the above problem.

    >>> np.random.seed(1)
    >>> X, Y, G, W = simulate_random(100, 6, 4)
    >>> Xh = X[:,G[0]]
    >>> wh = W[0]
    >>> gmax = G[3]
    >>> Xmax = X[:,gmax]
    >>> wmax = W[3]
    >>> Pmax = np.dot(X[:,gmax], X[:,gmax].T)
    >>> maximum_pinned(Xh, wh, X[:,gmax], wmax, Pmax, Y)
    1.0096735125191472

    """
    u_g = np.dot(X_g.T, y)
    u_g /= np.linalg.norm(u_g)
    
    a = np.dot(X_h.T, y - np.dot(P_g,y)) / w_h
    
    X_gu_g = np.dot(X_g, u_g)
    b = (w_g / w_h) * np.dot(X_h.T, X_gu_g) / (np.linalg.norm(X_gu_g)**2)
    
    a_dot_b = np.dot(a.T,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    assert np.linalg.norm(b) <= 1
    value = norm_a**2 / (np.sqrt(norm_a**2 - norm_a**2 * norm_b**2 + a_dot_b**2) - a_dot_b)
    return value

def test_statistic(X, Y, groups, weights):
    """
    The function computes the relevant quantities needed compute our $p$-values.

    Parameters
    ----------

    X : np.ndarray(np.float)
        Design matrix

    Y : np.ndarray(np.float)
        Outcome

    groups: [index_obj]
        Sequence of groups, each group being an index into the columns of X

    weights: [float]
        Sequence of weights for each group.

    Returns
    -------

    T : float
        Normalized gap between $\lambda_1$ and $\lambda_2$.
    
    L1: float
        $\lambda_1$, the first $\lambda$ for which the group LASSO 
        solution is non-zero.

    L2: float
        $\lambda_2$, the candidate $\lambda$ for when the next group is added.

    gmax : index_obj
        Group which achieves $\lambda_1$

    weight_gmax : float
        Weight of group that achieved $\lambda_1$

    rank : float
        Rank of $X_{g_{\max}}$ -- should round to an integer.

    >>> np.random.seed(1)
    >>> X, Y, G, W = simulate_random(100, 6, 4)
    >>> test_statistic(X,Y,G,W)
    (0.31521078773401379, 2.7386809827394636, 2.6235851429097683, slice(10, 15, None), 1.0000969506876749, 4.9999999999999982)
    >>> 

    """
    grad = np.dot(X.T, Y) 
    imax = np.argmax([np.linalg.norm(grad[group]) / weight for group, weight in zip(groups, weights)])
    gmax = groups[imax]
    X_gmax = X[:,gmax]
    P_gmax = np.dot(X_gmax, np.linalg.pinv(X_gmax))
    weight_gmax = weights[imax]
    M_u_gmax = L2 = max([maximum_pinned(X[:,group], weight, X_gmax, weight_gmax, P_gmax, Y) for group, weight in zip(groups, weights) if group != gmax])
    u_gmax = grad[gmax].copy()
    u_gmax /= np.linalg.norm(u_gmax)
    
    L1 = np.dot(u_gmax, grad[gmax]) / weight_gmax
    var_f_u_gmax = np.linalg.norm(np.dot(X_gmax, u_gmax))**2 # / (weight_gmax**2)
    T = L1 * (L1 - M_u_gmax) / var_f_u_gmax
    rank = np.diag(P_gmax).sum()
    return T, L1, L2, gmax, weight_gmax, rank

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
    Y = np.dot(X, beta) + np.random.standard_normal(n)
    weights = np.random.sample(g) + 1
    return X, Y, groups, weights
    
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


    soln_lasso, resid_lasso = solve_glasso(X_lasso, Y_lasso, np.arange(p), lagrange_lasso, tol=1.e-10)


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

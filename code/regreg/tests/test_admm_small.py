import regreg.api as rr, numpy as np
import regreg.knots as K
from regreg.knots.admm import admm

n, p = 100, 4
#np.random.seed(1)

X = np.random.standard_normal((n, p))
#X -= X.mean(0)[np.newaxis,:]
#X /= X.std(0)[np.newaxis,:]
#X /= np.sqrt(100)
Y = np.random.standard_normal((n,))

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

    alpha, var = K.find_alpha(soln, X, tangent_vectors)
    print alpha

    L = np.fabs(U).max()

    # having found alpha, we can 
    # solve for M+, M- explicitly (almost surely)
    Mplus = {}
    Mminus = {}
    keep = np.ones(U.shape[0], np.bool)
    keep[which] = 0

    den = 1 - alpha
    num = U - alpha * L
    Mplus[1] = (num / den * (den > 0))[keep]
    Mminus[1] = (num * keep / (den + (1 - keep)))[den < 0]
    
    den = 1 + alpha
    num =  -(U - alpha * L)
    Mplus[-1] = (num / den * (den > 0))[keep]
    Mminus[-1] = (num * keep / (den + (1 - keep)))[den < 0]
    
    print 'mplus', Mplus[1], Mplus[-1]

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
        
    return (L, Mplus, Mminus, alpha, tangent_vectors, var, U, alpha)

L, Mplus, Mminus, alpha, tangent_vectors, var, U, alpha = lasso_knot_covstat(X, Y, np.zeros(p))

epigraph = rr.l1_epigraph(p+1)

initial = np.zeros(p+1)
A = np.argmax(np.fabs(U))

for _ in range(1):
    #initial = np.random.standard_normal(p+1)
    print -K.linear_fractional_admm(-(U-alpha*L), alpha, epigraph, tol=1.e-10, rho=2*np.sqrt(p), initial=initial, min_iters=100), Mplus, L, np.sign(U[A])


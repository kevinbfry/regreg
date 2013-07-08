import numpy as np, nose.tools as nt

import regreg.api as rr
import regreg.knots as K
import regreg.knots.lasso as L


def sim_random_design(n, p):

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal((n,))

    sup_norm, Mcovstat, _, alpha, tangent_vectors, var, U, alpha = L.lasso_knot_covstat(X, Y, np.zeros(p))

    epigraph = rr.l1_epigraph(p+1)
    A = np.argmax(np.fabs(U))

    Madmm = -K.linear_fractional_admm(-(U-alpha*sup_norm), alpha, epigraph, tol=1.e-14, rho=np.sqrt(p), min_iters=2000, max_iters=3000)[0]

    print Mcovstat, Madmm, sup_norm, np.fabs(Mcovstat - Madmm) / Mplus
    return np.fabs(Mcovstat - Madmm) / Mplus

def test_admm_knot():

    np.random.seed(0)

    nt.assert_true(sim_random_design(100,10) < 1.e-2)
    nt.assert_true(sim_random_design(100,1000) < 1.e-2)
    nt.assert_true(sim_random_design(100,10000) < 1.e-2)

    

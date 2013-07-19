import numpy as np

import regreg.api as rr
from regreg.knots import Q_0

def pvalue(X, sigma=1, nsim=5000):
    n, p = X.shape
    D = np.linalg.svd(X)[1] / sigma
    m = n+p-2
    H = np.zeros(m)
    
    nonzero = np.hstack([D[1:],-D[1:]])
    H[:nonzero.shape[0]] = nonzero
        
    return min(Q_0(D[0], D[1], np.inf, H, nsim=nsim), 1)

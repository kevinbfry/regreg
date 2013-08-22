from cython.view cimport array as cvarray
import numpy as np
cimport numpy as cnp

DTYPE_float = np.float
ctypedef cnp.float_t DTYPE_float_t

def group_lasso_knot(cnp.ndarray[DTYPE_float_t, ndim=2] X,
                     double[:] Y,
                     long[:] groups,
                     double[:] weights):
    
    U = np.dot(X.T, Y)
    terms = np.zeros_like(weights) 
    cdef int i, j
    cdef float value
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            terms[groups[i]] = terms[groups[i]] + U[i]**2
    
    for j in range(weights.shape[0]):
        terms[j] = np.sqrt(terms[j]) / weights[j]
   
    cdef long imax = int(np.argmax(terms))
    L = terms[imax]
    
    # assuming groups are 0,1,2,...
    
    gmax = imax 
    wmax = weights[gmax]
    
    which = np.equal(groups, imax)
    
    # assuming rank == num of variables in group...
    kmax = which.sum()
    
    Uwhich = U[which]
    soln = (Uwhich / np.linalg.norm(Uwhich)) / wmax
    
    Xmax = np.array(X)[:,which]
    Xeta = np.dot(Xmax, soln)
    
    Xmax -= np.outer(Xeta, soln) / np.linalg.norm(soln)**2
    if kmax > 1:
        Wmax = Xmax[:,:-1]
        XetaP = np.dot(Wmax, np.linalg.lstsq(Wmax, Xeta)[0])
        Xeta -= XetaP

    conditional_variance = np.linalg.norm(Xeta)**2
    
    Xeta /= conditional_variance
    
    C_X = np.dot(X.T, Xeta)
    
    a = U - C_X * L
    b = C_X

    Vplus = []
    Vminus = []
    for label in xrange(weights.shape[0]):
        if label != gmax:
            group = np.equal(groups, label)
            weight = weights[label]
            tf = trignometric_form(a[group], b[group], weight)
            Vplus.append(tf[0])
            Vminus.append(tf[1])
    if Vplus:
        Mplus, Mminus = -np.nanmax(Vplus), np.nanmin(Vminus)
    else:
        Mplus, Mminus = 0, np.inf
    return L, -Mplus, Mminus, conditional_variance, kmax, wmax

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

    if np.linalg.norm(b) < w:
        return max([V1,V2]), np.inf
    else:
        return min([V1,V2]), max([V1,V2])

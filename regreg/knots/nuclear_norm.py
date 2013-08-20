import numpy as np, os

import regreg.api as rr
from regreg.knots import (find_C_X, 
                          linear_fractional_nesta, 
                          linear_fractional_tfocs,
                          linear_fractional_admm,
                          chi_pvalue,
                          Q_0)

def nuclear_norm_knot(X, R, 
                      epsilon=([1.e-2] + [1.e-4]*3 + [1.e-5]*3 + 
                               [1.e-6]*50 + [1.e-8]*200), tol=1.e-10, 
                      method='admm',
                      min_iters=200):
    """
    Find an approximate matrix completion knot
    """

    X = rr.astransform(X)
    Z = X.adjoint_map(R).copy()
    n, p = Z.shape
    U, D, V = np.linalg.svd(Z)
    soln = np.multiply.outer(U[:,0], V[0])
    L = np.max(D)

    tangent_vectors = ([np.multiply.outer(U[:,0], V[j]).reshape(-1) 
                        for j in range(1,p)]
                       + [np.multiply.outer(U[:,j], V[0]).reshape(-1) # copy necessary 
                          for j in range(1,n)])

    C_X, var = find_C_X(soln, X, tangent_vectors)
    
    G = np.zeros((n+p-2,n+p-2))
    neg_hessian = np.zeros((n+p-2,n+p-2))
    zero_vector = np.zeros((n,p))

#     for i in range(n-1):
#         for j in range(p-1):
#             G[i,n-1+j] = (U[:,i+1] * np.dot(C_X, V[j+1])).sum() 
#             G[n-1+j,i] = G[i,n-1+j]
#             neg_hessian[i,n-1+j] = (U[:,i+1] * np.dot(Z, V[j+1])).sum() 
#             neg_hessian[n-1+j,i] = neg_hessian[i,n-1+j]

    G[:(n-1),(n-1):] = np.dot(U[:,1:].T, np.dot(C_X, V[1:].T))
    G[(n-1):,:(n-1)] = G[:(n-1),(n-1):].T
    neg_hessian[:(n-1),(n-1):] = np.dot(U[:,1:].T, np.dot(Z, V[1:].T))
    neg_hessian[(n-1):,:(n-1)] = neg_hessian[:(n-1),(n-1):].T

    G += np.identity(n+p-2) * (U[:,0] * np.dot(C_X, V[0])).sum() 
    neg_hessian += np.identity(n+p-2) * L
#     for i in range(n+p-2):
#         G[i,i] = (U[:,0] * np.dot(C_X, V[0])).sum() 
#         neg_hessian[i,i] = L

    H = neg_hessian - G*L
    evalsG, evecsG = np.linalg.eigh(G)
    Ginvsqrt = np.dot(evecsG, (1./np.sqrt(evalsG)[:,np.newaxis]* evecsG.T))
    Hvals = np.linalg.eigvalsh(np.dot(Ginvsqrt, np.dot(H, Ginvsqrt)))

    # in actually finding M^+ we don't have to subtract the conditional 
    # expectation of the tangent part, as it will be zero at eta that
    # achieves L1

    epigraph = rr.nuclear_norm_epigraph((n,p))

    if method == 'nesta':
        Mplus, next_soln = linear_fractional_nesta(-(Z-C_X*L), 
                                                    C_X, 
                                                    epigraph, 
                                                    tol=tol,
                                                    epsilon=epsilon,
                                                    min_iters=min_iters)
    elif method == 'tfocs':
        Mplus, next_soln = linear_fractional_tfocs(-(Z-C_X*L), 
                                                    C_X, 
                                                    epigraph, 
                                                    tol=tol,
                                                    epsilon=epsilon,
                                                    min_iters=min_iters)
    elif method == 'admm':
        Mplus, next_soln = linear_fractional_admm(-(Z-C_X*L), 
                                                   C_X, 
                                                   epigraph, 
                                                   tol=tol,
                                                   rho=np.sqrt(max(n,p)),
                                                   min_iters=min_iters)
    elif method == 'explicit':
        Mplus, Mminus, next_soln = min(Hvals), np.inf, None
    else:
        raise ValueError('method must be one of ["nesta", "tfocs", "admm", "explicit"]')

    dual = rr.operator_norm((n,p), lagrange=1)

    if dual.seminorm(C_X) > 1.001:
        if method == 'nesta':
            Mminus, _ = linear_fractional_nesta(-(Z-C_X*L), 
                                                 C_X, 
                                                 epigraph, 
                                                 tol=tol,
                                                 sign=-1,
                                                 epsilon=epsilon,
                                                 min_iters=min_iters)
        elif method == 'tfocs':
            Mminus, _ = linear_fractional_tfocs(-(Z-C_X*L), 
                                                 C_X, 
                                                 epigraph, 
                                                 tol=tol,
                                                 sign=-1,
                                                 epsilon=epsilon,
                                                 min_iters=min_iters)
            
        elif method == 'admm':
            Mminus, _ = linear_fractional_admm(-(Z-C_X*L), 
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

    return (L, -Mplus, Mminus, C_X, tangent_vectors, var, U, next_soln, Hvals)

def first_test(X, Y, nsim=10000,
               method='explicit',
               sigma=1):
    X = rr.astransform(X)

    (L, Mplus, Mminus, _, _, 
     var, _, _, H) = nuclear_norm_knot(X, Y, 
                                       method=method,
                                       tol=1.e-12)
    sd = np.sqrt(var) * sigma

    pval = Q_0(L/sd, Mplus/sd, Mminus/sd, H/sd, nsim=nsim)

    if pval > 1:
        pval = 1

    return pval


def check_knots(nsim=50, seed=0):

    shape_nn = (100, 20)

    np.random.seed(seed)
    def solve_nuclear_norm(X, Y, L, tol=1.e-5, min_its=10):
        """
        Solve the nuclear norm problem with design matrix X, outcome Y
        and Lagrange parameter L.
        """
        loss = rr.squared_error(X, Y)
        penalty = rr.nuclear_norm(shape_nn, lagrange=L)
        problem = rr.simple_problem(loss, penalty)
        soln = problem.solve(tol=tol, min_its=min_its)
        resid = Y - X.linear_map(soln).copy()
        return soln, resid

    def find_next_knot_nn(X, R, soln, L, multiplicity, niter=50, verbose=False):

        loss = rr.squared_error(X, R)

        L2 = L
        for _ in range(niter):
            grad = loss.smooth_objective(soln, mode='grad')
            Lcandidate = (sorted(np.linalg.svd(grad)[1])[-(multiplicity+1)] + L2) / 2.
            if verbose:
                print sorted(np.linalg.svd(grad)[1])[-3:], Lcandidate
            soln = solve_nuclear_norm(X, R, Lcandidate, tol=1.e-12, min_its=100)[0]
            L2 = Lcandidate

        return L, L2

    if not os.path.exists('nuclear_norm_knots.npy'):
        values = []
        for i in range(nsim):
            observed_nn = np.random.binomial(1,0.3,shape_nn).astype(np.bool)
            X_nn = rr.selector(observed_nn, shape_nn)

            Z_nn = np.random.standard_normal(observed_nn.sum())
            U, D, V = np.linalg.svd(X_nn.adjoint_map(Z_nn))
            soln_nn = np.zeros(X_nn.input_shape)
            lagrange_nn = D.max()

            strong_rules_knot = find_next_knot_nn(X_nn, Z_nn, soln_nn, lagrange_nn, 1, verbose=False,
                                                  niter=40)[1]
            Mplus = nuclear_norm_knot(X_nn, Z_nn, 
                                      soln_nn,
                                      method='admm')[1]

            Mplus2 = nuclear_norm_knot(X_nn, Z_nn, 
                                      soln_nn,
                                      method='explicit')[1]

            values.append([Mplus, Mplus2, strong_rules_knot])
            print values[-1]

        values = np.array(values)
        np.save('nuclear_norm_knots.npy', np.array(values))
    else:
        values = np.load('nuclear_norm_knots.npy')

    from matplotlib import pyplot as plt

    plt.clf()
    plt.scatter(values[:,0], values[:,1], label=r'ADMM vs. explicit')
    plt.legend(loc='lower right')
    plt.savefig('nuclear_norm_knots1.pdf')

    plt.clf()
    plt.scatter(values[:,0], values[:,2], c='r', label=r'ADMM vs. $\lambda_2$')
    plt.legend(loc='lower right')
    plt.savefig('nuclear_norm_knots2.pdf')



import numpy as np, regreg.api as rr

def admm(atom1, atom2, 
         rho=1., 
         max_iters=1000, 
         tol=1.e-5, 
         coef_stop=False,
         min_iters=10):
    """
    Solve the problem:

    .. math::

       \text{minimize}_x f(x) + g(x)

    using ADMM, where $f$ is `atom1` and $g$ is `atom2`.

    Parameters
    ----------

    atom1, atom2: `regreg.atoms.atom`
         Objects having a `proximal` method.
    
    Following simple ADMM on p.14 of the 
    `ADMM paper <http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf>`_

    Specifically, we take $A=I, B=-I, c=0$.
    """

    if atom1.shape != atom2.shape:
        raise ValueError('atoms should have same shape!')

    shape = atom1.shape
    x, z = atom1.coefs, atom2.coefs
    y = np.random.standard_normal(atom1.shape)

    value = np.inf

    for iteration in range(max_iters):
        quad_x = rr.identity_quadratic(rho, z, y, 0)
        x = atom1.proximal(quad_x)

        quad_z = rr.identity_quadratic(rho, x, -y, 0)
        z = atom2.proximal(quad_z)

        y += rho * (x - z)

        updated_value = atom1.objective(z) + atom2.objective(z)
        if iteration > min_iters:
            if coef_stop and np.linalg.norm(x - z) < max(np.linalg.norm(x), 1) * tol:
                break
            elif np.fabs(value - updated_value) < max(np.fabs(value), 1) * tol:
                break

        value = updated_value

    return z

def test_admm():
    p = 100
    atom1 = rr.l1norm(p, lagrange=0.3)
    atom2 = rr.l1norm(p, lagrange=0.7)
    atom3 = rr.l1norm(p, lagrange=1)
    Z = np.random.standard_normal(p) * 10
    W = np.random.standard_normal(p) 
    quad1 = rr.identity_quadratic(1, Z, 0, 0)
    quad2 = rr.identity_quadratic(0, 0, W, 0)

    atom1.quadratic = quad1
    atom2.quadratic = quad2
    soln1 = admm(atom1, atom2, tol=1.e-7, rho=10.)
    soln2 = atom3.proximal(quad1 + quad2)
    print 'Z', Z
    print 'soln1', soln1
    print 'soln2', soln2
    print 'rel', np.linalg.norm(soln1 - soln2) / np.linalg.norm(soln1)
    print atom1.objective(soln1) + atom2.objective(soln1), atom1.objective(soln2) + atom2.objective(soln2)


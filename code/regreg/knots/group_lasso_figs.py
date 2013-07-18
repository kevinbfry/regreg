IP = get_ipython()
IP.magic('load_ext rmagic')

import regreg.knots.group_lasso as GL
import regreg.api as rr
from regreg.affine import fused_lasso 
import regreg.affine as ra
import numpy as np, random

import mpmath

def simulate_null(X, groups, weights={}, orthonormal=False):

    if orthonormal:
        for group in np.unique(groups):
            g = groups == group
            X[:,g] = np.linalg.svd(X[:,g], full_matrices=False)[0]

    sigma = 0.1
    n, p = X.shape

    Z = np.random.standard_normal(n) * sigma

    return GL.first_test(X, Z, groups, weights=weights, sigma=sigma)

def fig(X, fname, groups, nsim=10000, weights={}):
    P = []
    for _ in range(nsim):
        pval = simulate_null(X, groups, weights=weights)
        if pval is not None:
            P.append(pval)
    P = np.array(P)
    make_fig(fname, P)

def make_fig(fname, P):
    IP = get_ipython()
    IP.magic('load_ext rmagic')
    IP.magic('R -i P')
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname, P.shape[0]))

def fig1(nsim=10000):
    X = np.arange(12).reshape((3,4)) + 0.1 * np.random.standard_normal((3,4))
    groups = np.array([0,0,1,1])
    fig(X, 'small_group_lasso.pdf', groups, nsim=nsim, weights={0:0.1})

def fig2(nsim=10000):
    n, p = 100, 10000
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array(range(1000)*10)
    fig(X, 'fat_group_lasso.pdf', groups, nsim=nsim)

def fig3(nsim=10000):
    n, p = 10000, 100
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array(range(10)*10)
    fig(X, 'tall_group_lasso.pdf', groups, nsim=nsim)

def fig4(nsim=10000):
    n, p = 100, 100
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array(range(10)*10)
    fig(X, 'square_group_lasso.pdf', groups, nsim=nsim)

def fig5(nsim=10000):
    IP = get_ipython()
    IP.magic('load_ext rmagic')
    IP.run_cell_magic('R', '-o X', 
'''
library(lars)
data(diabetes)
X = diabetes$x
''')
    X = IP.user_ns['X']
    groups = [0,0,0,0,1,1,2,2,2,3]
    fig(X, 'lars_diabetes_group.pdf', groups, nsim=nsim, weights={2:1,3:2,0:4})

def fig6(nsim=10000):
    IP = get_ipython()
    IP.magic('load_ext rmagic')
    IP.run_cell_magic('R', '-o X', 
'''
library(lars)
data(diabetes)
X = diabetes$x
''')
    X = IP.user_ns['X']
    groups = range(X.shape[1])
    fig(X, 'lars_diabetes_lasso_as_group.pdf', groups, nsim=nsim, 
        weights=dict([(i, s) for i, s in enumerate(0.2 * np.random.sample(X.shape[1])+1)]))

def fig7(nsim=10000):
    n, p = 100, 10
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    X[:,8:] = X[:,:2]
    groups = np.array([0]*8+[1]*2)
    fig(X, 'nested_groups_big_first.pdf', groups, nsim=nsim, weights={0:0.1,1:3})

def fig8(nsim=10000):
    n, p = 100, 10
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    X[:,8:] = X[:,:2]
    groups = np.array([0]*8+[1]*2)
    fig(X, 'nested_groups_smaller_first.pdf', groups, nsim=nsim, weights={1:0.1,0:3})

def fig9(nsim=10000):
    n = 100
    Z = [np.random.standard_normal((n,4)) for _ in range(2)]
    X = np.hstack([Z[0], Z[0][:,:2], Z[1], Z[1][:,:2]])
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array([0]*4+[1]*2+[2]*4+[3]*2)
    fig(X, 'two_nested_groups.pdf', groups, nsim=nsim)

def fig10(nsim=10000):
    n = 100
    Z = [np.random.standard_normal((n,4)) for _ in range(20)]
    W = []
    groups = []
    for i in range(20):
        W.extend([Z[i], Z[i][:,:2]])
        groups.extend([2*i]*4+[2*i+1]*2)
    print groups
    X = np.hstack(W)
    X -= X.mean(0)
    X /= X.std(0)
    fig(X, 'several_nested_groups.pdf', groups, nsim=nsim, weights=
        {0:1.9,2:1.9,5:1.3})

def produce_figs(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]]

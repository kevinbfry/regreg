IP = get_ipython()
IP.magic('load_ext rmagic')

import os
import regreg.knots.group_lasso as GL
import regreg.api as rr
from regreg.affine import fused_lasso 
import regreg.affine as ra
import numpy as np, random

import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

df = 5
sigma = (np.random.standard_normal(10000) / np.sqrt(np.random.chisquare(df, size=(10000,)) / df)).std()

def simulate_null(X, groups, weights={}, orthonormal=False):

    if orthonormal:
        for group in np.unique(groups):
            g = groups == group
            X[:,g] = np.linalg.svd(X[:,g], full_matrices=False)[0]

    n = X.shape[0]

    Z1 = np.random.standard_normal(n) / np.sqrt(np.random.chisquare(df, size=(n,)) / df)
    Z2 = (np.random.exponential(1, size=(n,)) - 1.) * 3
    Z = Z1+Z2

    return GL.first_test(X, Z, groups, weights=weights, sigma=np.sqrt(sigma**2+9))

def fig(X, fname, groups, nsim=10000, weights={}):
    P = []
    for i in range(nsim):
        pval = simulate_null(X, groups, weights=weights)
        if pval is not None:
            P.append(pval)
        if i % 1000 == 0:
            dname = os.path.splitext(fname)[0] + '.npy'
            np.save(dname, np.array(P))
    P = np.array(P)
    make_fig(fname, P)

def make_fig(fname, Pv):

    IP = get_ipython()
    IP.magic('load_ext rmagic')

    P = Pv[:,0]
    IP.magic('R -i P')
    dname = os.path.splitext(fname)[0] + '.npy'
    np.save(dname, Pv)

    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname, P.shape[0]))

    P = Pv[:,1]
    IP.magic('R -i P')
    dname = os.path.splitext(fname)[0] + '.npy'
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname.replace('.pdf', '_exp.pdf'), P.shape[0]))

def fig2(nsim=10000):
    n, p = 100, 10000
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array(range(1000)*10)
    fig(X, 'fat_group_lasso_texp.pdf', groups, nsim=nsim)

def fig3(nsim=10000):
    n, p = 10000, 100
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array(range(10)*10)
    fig(X, 'tall_group_lasso_texp.pdf', groups, nsim=nsim)

def fig4(nsim=10000):
    n, p = 100, 100
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array(range(10)*10)
    fig(X, 'square_group_lasso_texp.pdf', groups, nsim=nsim)

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
    fig(X, 'lars_diabetes_group_texp.pdf', groups, nsim=nsim, weights={2:1,3:2,0:4})

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
    fig(X, 'lars_diabetes_lasso_as_group_texp.pdf', groups, nsim=nsim, 
        weights=dict([(i, s) for i, s in enumerate(0.2 * np.random.sample(X.shape[1])+1)]))

def fig7(nsim=10000):
    n, p = 100, 10
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    X[:,8:] = X[:,:2]
    groups = np.array([0]*8+[1]*2)
    fig(X, 'nested_groups_big_first_texp.pdf', groups, nsim=nsim, weights={0:0.1,1:3})

def fig8(nsim=10000):
    n, p = 100, 10
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    X[:,8:] = X[:,:2]
    groups = np.array([0]*8+[1]*2)
    fig(X, 'nested_groups_smaller_first_texp.pdf', groups, nsim=nsim, weights={1:0.1,0:3})

def fig9(nsim=10000):
    n = 100
    Z = [np.random.standard_normal((n,4)) for _ in range(2)]
    X = np.hstack([Z[0], Z[0][:,:2], Z[1], Z[1][:,:2]])
    X += np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    groups = np.array([0]*4+[1]*2+[2]*4+[3]*2)
    fig(X, 'two_nested_groups_texp.pdf', groups, nsim=nsim)

def fig10(nsim=10000):
    n = 100
    Z = [np.random.standard_normal((n,4)) for _ in range(20)]
    W = []
    groups = []
    for i in range(20):
        W.extend([Z[i], Z[i][:,:2]])
        groups.extend([2*i]*4+[2*i+1]*2)
    X = np.hstack(W)
    X += np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    fig(X, 'several_nested_groups_texp.pdf', groups, nsim=nsim, weights=
        {0:1.9,2:1.9,5:1.3})




def produce_figs(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig2, fig3, fig4, fig5, 
                   fig6, fig7, fig8, fig9, fig10]]

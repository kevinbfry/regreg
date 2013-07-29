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
    X += np.random.standard_normal(n)[:,np.newaxis]
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
    X = np.hstack(W)
    X += np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    fig(X, 'several_nested_groups.pdf', groups, nsim=nsim, weights=
        {0:1.9,2:1.9,5:1.3})


def fig11():
    plt.clf()

    P = []
    for f in ['several_nested_groups.npy',
              'small_group_lasso.npy',
              #'fat_group_lasso.npy',
              #'tall_group_lasso.npy',
              'square_group_lasso.npy',
              'lars_diabetes_group.npy',
              'lars_diabetes_lasso_as_group.npy',
              'nested_groups_big_first.npy',
              'nested_groups_smaller_first.npy',
              'two_nested_groups.npy',
              'several_nested_groups.npy']:
        a = np.load(f)
        if a.ndim == 2:
            P.append(a[:,0])
        else:
            P.append(a)
    P = np.asarray(P).reshape(-1)
    np.random.shuffle(P)
    P = P[:20000]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, linewidth=4, color='red')

    plt.plot([0,1],[0,1], '--', linewidth=2, color='black')
    plt.savefig('group_lasso_pval_ecdf.pdf')

def fig12():
    plt.clf()

    P = np.load('small_group_lasso.npy')[:,1]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, label=r'$3\times 4$', linewidth=2)

    P = np.load('lars_diabetes_group.npy')[:,1]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, label=r'diabetes', linewidth=2)

    P = np.load('several_nested_groups.npy')[:,1]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, label=r'nested groups', linewidth=2)

    plt.plot([0,1],[0,1], '--', linewidth=1, color='black')
    plt.legend(loc='upper left')
    plt.savefig('group_lasso_exp_ecdf.pdf')



def produce_figs(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig3, fig4, fig5, 
                   fig6, fig7, fig8, fig9, fig10]]

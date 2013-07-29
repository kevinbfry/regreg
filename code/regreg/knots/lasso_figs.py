import os
import regreg.knots.lasso as K
from regreg.affine import fused_lasso 
import regreg.affine as ra
import numpy as np, random

import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

def simulate_null(X):
    sigma = 0.1
    n, p = X.shape

    Z = np.random.standard_normal(n) * sigma
    return K.first_test(X, Z, sigma=sigma)

def fig(X, fname, nsim=10000):
    IP = get_ipython()
    P = []
    for i in range(nsim):
        if i % 1000 == 0:
            dname = os.path.splitext(fname)[0] + '.npy'
            np.save(dname, np.array(P))
        P.append(simulate_null(X))
    Pv = np.array(P)

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
''' % (fname, nsim))

    P = Pv[:,1]
    IP.magic('R -i P')
    dname = os.path.splitext(fname)[0] + '.npy'
    np.save(dname, Pv)
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname.replace('.pdf', '_exp.pdf'), nsim))

def fig1(nsim=10000):
    X = np.arange(6).reshape((3,2))
    fig(X, 'small_lasso.pdf', nsim=nsim)

def fig2(nsim=10000):
    n, p = 100, 10000
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    fig(X, 'fat_lasso.pdf', nsim=nsim)

def fig3(nsim=10000):
    n, p = 10000, 100
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis]
    X -= X.mean(0)
    X /= X.std(0)
    fig(X, 'tall_lasso.pdf', nsim=nsim)

def fig4(nsim=10000):
    n = 500
    D = fused_lasso.trend_filter(n)
    X = ra.todense(D)
    fig(X, 'fused_lasso.pdf', nsim=nsim)

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
    fig(X, 'lars_diabetes.pdf', nsim=nsim)

def fig6():
    plt.clf()

    P = np.load('small_lasso.npy')[:,1]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, label=r'$3\times 2$', linewidth=2)

    P = np.load('fused_lasso.npy')[:,1]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, label=r'fused LASSO', linewidth=2)

    P = np.load('lars_diabetes.npy')[:,1]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, label=r'diabetes', linewidth=2)

    plt.plot([0,1],[0,1], '--', linewidth=1, color='black')
    plt.legend(loc='upper left')
    plt.savefig('lasso_exp_ecdf.pdf')


def fig7():
    plt.clf()

    P = []
    for f in ['small_lasso.npy',
              'fused_lasso.npy',
              'fat_lasso.npy',
              'tall_lasso.npy',
              'lars_diabetes.npy']:
        P.append(np.load(f)[:,0])
    P = np.asarray(P).reshape(-1)
    np.random.shuffle(P)
    P = P[:20000]
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P), P.shape[0])
    y = ecdf(x)
    plt.step(x, y, linewidth=4, color='red')

    plt.plot([0,1],[0,1], '--', linewidth=2, color='black')
    plt.savefig('lasso_pval_ecdf.pdf')



def produce_figs(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig3, fig4, fig5]]

import regreg.knots.lasso as K
import regreg.api as rr
from regreg.affine import fused_lasso 
import regreg.affine as ra
from scipy.stats import chi
import numpy as np, random

def simulate_null(X):
    sigma = 0.1
    n, p = X.shape

    Z = np.random.standard_normal(n) * sigma
    return K.first_test(X, Z, sigma=sigma)

def fig(X, fname, nsim=10000):
    IP = get_ipython()
    P = []
    for _ in range(nsim):
        P.append(simulate_null(X))
    IP.magic('load_ext rmagic')
    IP.magic('R -i P')
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname, nsim))

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

def produce_figs(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig3, fig4, fig5]]

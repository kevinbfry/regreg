import regreg.knots.pca as PCA
import numpy as np, random

def simulate_null(n,p):
    sigma = 0.1
    X = np.random.standard_normal((n,p)) * sigma
    if min(n,p) < 10:
        nsim = 50000
    else:
        nsim = 5000
    return PCA.pvalue(X, sigma=sigma, nsim=nsim)

def fig(shape, fname, nsim=10000):
    IP = get_ipython()
    P = []
    for _ in range(nsim):
        P.append(simulate_null(*shape))
    IP.magic('load_ext rmagic')
    IP.magic('R -i P')
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname, nsim))

def fig1(nsim=10000):
    fig((3,4), 'small_pca.pdf', nsim=nsim)

def fig2(nsim=10000):
    fig((50,50), 'square_pca.pdf', nsim=nsim)

def fig3(nsim=10000):
    fig((30,1000), 'fat_pca.pdf', nsim=nsim)

def fig4(nsim=10000):
    fig((2,2), 'tiny_pca.pdf', nsim=nsim)

def fig5(nsim=10000):
    fig((30,5), 'medium_pca.pdf', nsim=nsim)

def fig6(nsim=10000):
    fig((100,20), 'midrange_pca.pdf', nsim=nsim)

def fig7(nsim=10000):
    fig((1000,1000), 'bigsquare_pca.pdf', nsim=nsim)

def produce_figs(seed=0, big=False):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig3, fig4, fig5, fig6]]
    if big:
        fig7()

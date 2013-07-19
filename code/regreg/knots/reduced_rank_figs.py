import regreg.api as rr
import regreg.knots.matrix_completion as MC
import numpy as np, random

def simulate_null(n, p, q):
    sigma = 0.1
    X = rr.linear_transform(np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis])
    X.input_shape = (p, q)
    X.output_shape = (n, q)
    Y = np.random.standard_normal((n,q)) * sigma
    nsim = 10000
    return MC.first_test(X, Y, sigma=sigma, nsim=nsim)

def fig(n, p, q, fname, nsim=10000):
    IP = get_ipython()
    P = []
    for _ in range(nsim):
        P.append(simulate_null(n, p, q))
        print np.mean(P), np.std(P), len(P)
    IP.magic('load_ext rmagic')
    IP.magic('R -i P')
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname, nsim))

def fig1(nsim=3000):
    fig(100, 10, 5, 'small_reduced_rank.pdf', nsim=nsim)

def produce_figs(seed=0, big=False):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1]]
    if big:
        fig3()

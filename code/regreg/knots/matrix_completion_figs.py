import regreg.api as rr
import regreg.knots.nuclear_norm as NN
import numpy as np, random

def simulate_null(data):
    if type(data) == type(()): # it is a (proportion, shape) tuple
        proportion, shape = data
        observed = np.random.binomial(1, proportion,shape).astype(np.bool)
    else:  # it is a boolean indicator matrix
        observed = data

    sigma = 0.1
    Y = np.random.standard_normal(observed.sum()) * sigma
    X = rr.selector(observed, observed.shape)
    nsim = 10000
    return NN.first_test(X, Y, sigma=sigma, nsim=nsim)

def fig(data, fname, nsim=10000):
    IP = get_ipython()
    P = []
    for _ in range(nsim):
        P.append(simulate_null(data))
        print np.mean(P), np.std(P), len(P)
    IP.magic('load_ext rmagic')
    IP.magic('R -i P')
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
abline(0,1, lwd=3, lty=2)
dev.off()
''' % (fname, nsim))

def fig1(nsim=10000):
    observed = np.ones((3,4), np.bool)
    fig(observed, 'small_matrixcomp_full.pdf', nsim=nsim)

def fig2(nsim=10000):
    shape = (10,5)
    observed = np.random.binomial(1,0.5,shape).astype(np.bool)
    fig(observed, 'small_matrixcomp.pdf', nsim=nsim)

def fig3(nsim=3000):
    shape = (100,30)
    observed = np.random.binomial(1,0.2,shape).astype(np.bool)
    fig(observed, 'medium_matrixcomp.pdf', nsim=nsim)

def fig4(nsim=10000):
    shape = (10,5)
    observed = np.ones(shape, np.bool)
    for i in range(5):
        observed[i,i] = 0
        observed[5+i,i] = 0
    fig(observed, 'deterministic1_matrixcomp.pdf', nsim=nsim)

def fig5(nsim=10000):
    shape = (20,10)
    observed = np.ones(shape, np.bool)
    for i in range(10):
        observed[i,i] = 0
        observed[10+i,i] = 0
    observed[0,:-1] = 0
    fig(observed, 'deterministic2_matrixcomp.pdf', nsim=nsim)

def fig6(nsim=10000):
    shape = (10,5)
    fig((0.7, shape), 'small_matrixcomp_random.pdf', nsim=nsim)

def produce_figs(seed=0, big=False):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig4, fig5, fig6]]
    if big:
        fig3()

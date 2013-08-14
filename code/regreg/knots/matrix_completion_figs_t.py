import regreg.api as rr
import regreg.knots.nuclear_norm as NN
import numpy as np, random, os

df = 5

def simulate_null(data):
    if type(data) == type(()): # it is a (proportion, shape) tuple
        proportion, shape = data
        observed = np.random.binomial(1, proportion, shape).astype(np.bool)
    else:  # it is a boolean indicator matrix
        observed = data


    n = observed.sum()
    Y = np.random.standard_normal(n) / np.sqrt(np.random.chisquare(df, size=(n,)) / df)
    sigma = (np.random.standard_normal(10000) / np.sqrt(np.random.chisquare(df, size=(10000,)) / df)).std()

    X = rr.selector(observed, observed.shape)
    nsim = 10000
    return NN.first_test(X, Y, sigma=sigma, nsim=nsim)

def fig(data, fname, nsim=10000, output_cycle=5000):
    IP = get_ipython()
    P = []
    for i in range(nsim):
        P.append(simulate_null(data))
        print np.mean(P), np.std(P), len(P)

        if i % output_cycle == 0 and i > 0:
            dname = os.path.splitext(fname)[0] + '.npy'
            np.save(dname, np.array(P))

            IP.magic('load_ext rmagic')
            IP.magic('R -i P')
            IP.run_cell_magic(u'R', u'', '''
        pdf('%s')
        qqplot(P, runif(%d), xlab='P-value', ylab='Uniform', pch=23, cex=0.5, bg='red')
        abline(0,1, lwd=3, lty=2)
        dev.off()
        ''' % (fname, nsim))

    dname = os.path.splitext(fname)[0] + '.npy'
    np.save(dname, P)
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
    fig(observed, 'small_matrixcomp_full_t.pdf', nsim=nsim,
        output_cycle=nsim)

def fig2(nsim=10000):
    shape = (10,5)
    observed = np.random.binomial(1,0.5,shape).astype(np.bool)
    fig(observed, 'small_matrixcomp_t.pdf', nsim=nsim,
        output_cycle=1000)

def fig3(nsim=10000):
    shape = (100,30)
    observed = np.random.binomial(1,0.8,shape).astype(np.bool)
    fig(observed, 'medium_matrixcomp_t.pdf', nsim=nsim,
        output_cycle=1000)

def fig4(nsim=10000):
    shape = (10,5)
    observed = np.ones(shape, np.bool)
    for i in range(5):
        observed[i,i] = 0
        observed[5+i,i] = 0
    fig(observed, 'deterministic1_matrixcomp_t.pdf', nsim=nsim,
        output_cycle=5000)

def fig5(nsim=10000):
    shape = (20,10)
    observed = np.ones(shape, np.bool)
    for i in range(10):
        observed[i,i] = 0
        observed[10+i,i] = 0
    observed[0,:-1] = 0
    fig(observed, 'deterministic2_matrixcomp_t.pdf', nsim=nsim,
        output_cycle=5000)

def fig6(nsim=10000):
    shape = (10,5)
    fig((0.7, shape), 'small_matrixcomp_random_t.pdf', nsim=nsim,
        output_cycle=5000)

def fig7(nsim=10000):
    shape = (200,100)
    observed = np.random.binomial(1,0.1,shape).astype(np.bool)
    fig(observed, 'larger_matrixcomp_t.pdf', nsim=nsim,
        output_cycle=100)

def fig8(nsim=10000):
    shape = (200,100)
    observed = np.random.binomial(1,0.1,shape).astype(np.bool)
    fig((0.1, shape), 'larger_matrixcomp_random_t.pdf', nsim=nsim,
        output_cycle=100)


def produce_figs(seed=0, big=False):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig4, fig5, fig6]]
    if big:
        fig3(); fig7(); fig8()

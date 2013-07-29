import regreg.api as rr
import regreg.knots.nuclear_norm as NN
import numpy as np, random, os

def simulate_null(n, p, q, method='explicit'):
    sigma = 0.1
    X = rr.linear_transform(np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,np.newaxis])
    X.input_shape = (p, q)
    X.output_shape = (n, q)
    Y = np.random.standard_normal((n,q)) * sigma
    nsim = 10000
    return NN.first_test(X, Y, sigma=sigma, nsim=nsim, method=method)

def fig(n, p, q, fname, nsim=10000, method='explicit'):
    IP = get_ipython()
    P = []
    for i in range(nsim):
        P.append(simulate_null(n, p, q, method=method))
        print np.mean(P), np.std(P), len(P)

        if i % 1000 == 0:
            dname = os.path.splitext(fname)[0] + '.npy'
            np.save(dname, np.array(P))

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

def fig1(nsim=10000, method='explicit'):
    fig(100, 10, 5, 'small_reduced_rank.pdf', nsim=nsim, method=method)

def fig2(nsim=10000, method='explicit'):
    fig(100, 20, 10, 'bigger_reduced_rank.pdf', nsim=nsim, method=method)

def fig3(nsim=10000, method='explicit'):
    fig(10, 100, 5, 'fat_reduced_rank.pdf', nsim=nsim, method=method)

def produce_figs(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig3]]


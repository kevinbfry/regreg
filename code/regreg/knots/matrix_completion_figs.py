import regreg.api as rr
import regreg.knots.nuclear_norm as NN
import numpy as np, random, os

def simulate_null(data):
    if type(data) == type(()): # it is a (proportion, shape) tuple
        proportion, shape = data
        observed = np.random.binomial(1, proportion, shape).astype(np.bool)
    else:  # it is a boolean indicator matrix
        observed = data

    sigma = 0.1
    Y = np.random.standard_normal(observed.sum()) * sigma
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
    fig(observed, 'small_matrixcomp_full.pdf', nsim=nsim,
        output_cycle=nsim)

def fig2(nsim=10000):
    shape = (10,5)
    observed = np.random.binomial(1,0.5,shape).astype(np.bool)
    fig(observed, 'small_matrixcomp.pdf', nsim=nsim,
        output_cycle=1000)

def fig3(nsim=10000):
    shape = (100,30)
    observed = np.random.binomial(1,0.2,shape).astype(np.bool)
    fig(observed, 'medium_matrixcomp.pdf', nsim=nsim,
        output_cycle=1000)

def fig4(nsim=10000):
    shape = (10,5)
    observed = np.ones(shape, np.bool)
    for i in range(5):
        observed[i,i] = 0
        observed[5+i,i] = 0
    fig(observed, 'deterministic1_matrixcomp.pdf', nsim=nsim,
        output_cycle=5000)

def fig5(nsim=10000):
    shape = (20,10)
    observed = np.ones(shape, np.bool)
    for i in range(10):
        observed[i,i] = 0
        observed[10+i,i] = 0
    observed[0,:-1] = 0
    fig(observed, 'deterministic2_matrixcomp.pdf', nsim=nsim,
        output_cycle=5000)

def fig6(nsim=10000):
    shape = (10,5)
    fig((0.7, shape), 'small_matrixcomp_random.pdf', nsim=nsim,
        output_cycle=5000)

def fig7(nsim=10000):
    shape = (200,100)
    observed = np.random.binomial(1,0.1,shape).astype(np.bool)
    fig(observed, 'larger_matrixcomp.pdf', nsim=nsim,
        output_cycle=100)

def fig8(nsim=10000):
    shape = (200,100)
    observed = np.random.binomial(1,0.1,shape).astype(np.bool)
    fig((0.1, shape), 'larger_matrixcomp_random.pdf', nsim=nsim,
        output_cycle=100)

def fig9(nsim=10000):
    shape = (20,10)
    observed = np.ones(shape, np.bool)

    for i in range(10):
        observed[i,i] = 0
        observed[10+i,i] = 0
    observed[0,:-1] = 0

    sigma = 0.1
    X = rr.selector(observed, observed.shape)

    L = []
    P = []
    for i in range(nsim):
        Y = np.random.standard_normal(observed.sum()) * sigma
        A = X.adjoint_map(Y)
        lam_max = np.linalg.svd(A)[1].max()
        L.append(lam_max)
        Z = np.random.standard_normal((50,50))
        P.append(np.linalg.svd(Z)[1].max())
        if i % 1000 == 0:
            print 'completed %d' % i

    L = np.array(L)**2
    L -= L.mean()
    L /= L.std()

    P = np.array(P)**2
    P -= P.mean()
    P /= P.std()

    IP = get_ipython()
    IP.magic('load_ext rmagic')
    IP.magic('R -i L,P')
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
plot(density(L), lwd=2, col='red')
lines(density(P), lwd=1,lty=2)
dev.off()
pdf('%s')
qqplot(L, P)
''' % ('deterministic2_lammax.pdf', 'deterministic2_lammax_qq.pdf'))


def fig10(nsim=10000):
    shape = (20,10)
    observed = np.ones(shape, np.bool)

    for i in range(10):
        observed[i,i] = 0
        observed[10+i,i] = 0
    observed[0,:-1] = 0

    sigma = 0.1
    X = rr.selector(observed, observed.shape)

    L = []
    P = []
    for i in range(nsim):
        Y = np.random.standard_normal(observed.sum()) * sigma
        L1, Mplus = NN.nuclear_norm_knot(X, Y)[:2]
        L.append(L1-Mplus)
        Z = np.random.standard_normal((50,50))
        L2, L1 = sorted(np.linalg.svd(Z)[1])[-2:]
        P.append(L1-L2)
        if i % 100 == 0:
            print 'completed %d' % i

    Q = np.array([L,P])
    np.save('deterministic2_gap.npy', Q)

    L = np.array(L)**2
    L -= L.mean()
    L /= L.std()

    P = np.array(P)**2
    P -= P.mean()
    P /= P.std()

    IP = get_ipython()
    IP.magic('load_ext rmagic')
    IP.magic('R -i L,P')
    IP.run_cell_magic(u'R', u'', '''
pdf('%s')
plot(density(L), lwd=2, col='red')
lines(density(P), lwd=1,lty=2)
dev.off()
pdf('%s')
qqplot(L, P)
dev.off()
pdf('%s')
mL = L - min(L)
mP = P - min(P)
qqplot(exp(-mL), runif(length(mL)))
dev.off()
''' % ('deterministic2_gap.pdf', 'deterministic2_gap_qq.pdf', 'deterministic2_gap_qq_exp.pdf'))


def produce_figs(seed=0, big=False):
    np.random.seed(seed)
    random.seed(seed)
    IP = get_ipython()
    IP.magic('R set.seed(%d)' % seed)

    [f() for f in [fig1, fig2, fig4, fig5, fig6]]
    if big:
        fig3(); fig7(); fig8()

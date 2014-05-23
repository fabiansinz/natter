import sys
sys.path.append('../')

from natter.DataModule import DataLoader
from natter.Transforms import LinearTransformFactory
from natter.Distributions import ProductOfExponentialPowerDistributions, LpSphericallySymmetric, CompleteLinearModel,  LpGeneralizedNormal, Distribution, Histogram
from natter.DataModule import Data
from collections import defaultdict
from numpy import *
import itertools
from matplotlib.pyplot import *
from IPython import embed
from collections import OrderedDict


def disjoin_axes(ax):
    ax.tick_params(('direction','out'))
    for loc, spine in ax.spines.iteritems():
        if loc in ['left']:
            spine.set_position(('outward',10.))
        elif loc in ['bottom']:
            spine.set_position(('outward',10.))
        elif loc in ['right','top']:
            spine.set_color('none') # don't draw spine

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def box_off(ax):
    ax.tick_params(('direction','out'))
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    

if len(sys.argv) == 1:
    print """Usage: python code.py EXAMPLENO
             The example number determines which example from the paper is executed.
             1: model comparison example
             2: implementing new distributions (uniform distributions in the Lp-unit ball)
             3: Nonlinear transforms and log-determinants"""

#================= model comparison code ===========================================
elif int(sys.argv[1]) == 1:


    #-- load data
    print "Loading a simple Data module from an ascii file"
    FILENAME_TRAIN = 'hateren8x8_train_No1.dat.gz'
    FILENAME_TEST = 'hateren8x8_test_No1.dat.gz'

    dat_train = DataLoader.load(FILENAME_TRAIN)
    dat_test = DataLoader.load(FILENAME_TEST)

    #-- center and scale data 

    mu_train = dat_train.center()
    dat_test.center(mu_train)

    s = dat_train.makeWhiteningVolumeConserving()
    dat_test.makeWhiteningVolumeConserving(D=s)

    #-- get AC and whitening filters 

    FDCAC = LinearTransformFactory.DCAC(dat_train)
    FDC = FDCAC[0,:]
    FAC = FDCAC[1:,:]
    FwPCA = LinearTransformFactory.wPCA(FAC*dat_train)

    dat_train = FwPCA*FAC*dat_train
    dat_test = FwPCA*FAC*dat_test

    #-- Creating the distribution objects
    n = dat_train.dim()

    q = LpSphericallySymmetric(n=n,p=1.3)
    q.primary.remove('p')

    models = OrderedDict([
        ('factorial exponential power', ProductOfExponentialPowerDistributions(n=n)),
        (r'$p$-generalized Normal', LpGeneralizedNormal(n=n)),
        (r'$L_p$-spherical', LpSphericallySymmetric(n=n)),
        ('complete linear model', CompleteLinearModel(n=n,q=q,
                                    W=LinearTransformFactory.fastICA(dat_train)))
        ])

    # -- fitting and testing the distributions
    avg_log_loss = defaultdict(list)

    for model,p in models.items():
        p.estimate(dat_train[:,:5000])
        for xtest in dat_test.bootstrap(50,dat_test.numex()):
            avg_log_loss[model].append(p.all(xtest))

    fig = figure()
    ax = fig.add_subplot(111)
    K = ['factorial exponential power', r'$p$-generalized Normal',\
         r'$L_p$-spherical', 'complete linear model' ]
    bp = ax.boxplot([avg_log_loss[k] for k in K],positions=arange(len(K)))
    fig.subplots_adjust(bottom=.15,hspace=.25,wspace=.25)
    disjoin_axes(ax)
    ax.set_xticks(range(len(K)))
    ax.set_xticklabels(K)
    ax.set_title('Model Likelihood',fontweight='bold', fontsize=14)
    ax.set_xlabel('Models',fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Log-Loss [Bits/Component]',fontweight='bold', fontsize=12)

    for v in itertools.chain(*bp.values()):
        v.set_linewidth(1.5)

    #-- plotting two-dimensional log-contours

    fig = figure()

    for i,(model,p) in enumerate(models.items()):
        ax = fig.add_subplot(2,2,i+1)
        dat2 = p.sample(300000)
        dat_train[:2,:].plot(ax=ax,plottype='loghist',colors='b',label='true data')
        dat2[:2,:].plot(ax=ax,plottype='loghist',colors='r',label='sampled data')
        ax.set_title(model, fontweight='bold',fontsize=14)
        ax.axis('off')
        ax.legend(frameon=False)



    #-- plotting radial distributions
    fig = figure()

    for i,model in enumerate([r'$p$-generalized Normal', r'$L_p$-spherical']):
        p = models[model]

        ax = fig.add_subplot(2,1,i+1)
        p['rp'].histogram(dat_test.norm(p['p']), bins=100, ax=ax)
        ax.set_title(r'radial distribution of ' + model, fontsize=14,fontweight='bold')
        ax.set_xlabel('||x||',fontweight='bold',fontsize=12)
        ax.set_ylabel('p(||x||)',fontweight='bold',fontsize=12)
    fig.subplots_adjust(hspace=.3,wspace=.3)


    #-- plotting filters
    fig = figure()

    ax =fig.add_subplot(111)
    F = FDC.stack(models['complete linear model']['W']*FwPCA*FAC)
    F.plotFilters(ax=ax)




    show()

#================= implementing new distributions =================================
elif int(sys.argv[1]) == 2:
    from natter.Auxiliary.Utils import parseParameters
    from scipy import stats
    from scipy.special import digamma, beta

    class Beta(Distribution):


        def __init__(self, *args, **kwargs):
            param = parseParameters(args,kwargs)
        
            # set default parameters
            self.name = 'Beta Distribution'
            self.param = {'alpha':1.0,'beta':1.0}

            if param is not None:
                for k in param.keys():
                    self.param[k] = float(param[k])
            self.primary = ['alpha','beta']

        def pdf(self, dat):
            return squeeze(stats.beta.pdf(dat.X,self['alpha'], self['beta'] ))

        def loglik(self,dat):
            return log(self.pdf(dat))

        def sample(self,m):
            return Data(stats.beta.rvs(self['alpha'], self['beta'],size=(m,)))

        def primary2array(self):
            ret = zeros(len(self.primary))
            for ind,key in enumerate(self.primary):
                ret[ind]=self.param[key]
            return ret

        def array2primary(self, arr):
            ind = 0
            for ind, key in enumerate(self.primary):
                self.param[key] = arr[ind]
            return self

        def primaryBounds(self):
            return len(self.primary)*[(1e-6,None)]


        def dldtheta(self, dat):
            ret = zeros((len(self.primary), dat.numex()))
            x = dat.X[0]
            a = self['alpha']
            b = self['beta']
            p = self.pdf(dat)
            for ind, key in enumerate(self.primary):
                if key is 'alpha':
                    ret[ind,:] = p*(digamma(a+b)-digamma(a)+log(x))
                elif key is 'beta':
                    ret[ind,:] = p*(digamma(a+b)-digamma(b)+log(1-x))
            return ret
                    
            


    p_true = LpSphericallySymmetric(n=2,rp=Beta(alpha=2. , beta=1.),p=.5)
    p_est = LpSphericallySymmetric(n=2,rp=Beta(alpha=2,beta=2),p=.5)
    p_est.primary.remove('p')
    p_unest = p_est.copy()


    dat = p_true.sample(5000)
    p_est.estimate(dat)
    dat_false = p_unest.sample(5000)
    dat_est = p_est.sample(5000)
    
    fig = figure()
    ax = fig.add_subplot(231)
    dat.plot(ax=ax)
    ax.axis([-1,1,-1,1])
    disjoin_axes(ax)
    ax.set_xlabel(r'$x_1$',fontsize=15)
    ax.set_ylabel(r'$x_2$',fontsize=15)
    ax.set_title('true distribution',fontsize=15,fontweight='bold')
    

    ax = fig.add_subplot(232)
    dat_false.plot(ax=ax)
    ax.axis([-1,1,-1,1])
    disjoin_axes(ax)
    ax.set_xlabel(r'$x_1$',fontsize=15)
    ax.set_ylabel(r'$x_2$',fontsize=15)
    ax.set_title('before estimation',fontsize=15,fontweight='bold')

    ax = fig.add_subplot(233)
    dat_est.plot(ax=ax)
    ax.axis([-1,1,-1,1])
    disjoin_axes(ax)
    ax.set_xlabel(r'$x_1$',fontsize=15)
    ax.set_ylabel(r'$x_2$',fontsize=15)
    ax.set_title('after estimation',fontsize=15,fontweight='bold')

    ax = fig.add_subplot(234)
    p_true['rp'].histogram(dat.norm(p=p_true['p']),ax=ax)
    ax.set_ylim((0,5))
    ax.set_xlabel(r'$||x||_p$',fontsize=15)
    ax.set_ylabel(r'$\rho(||x||_p)$',fontsize=15)
    box_off(ax)
    
    ax = fig.add_subplot(235)
    p_unest['rp'].histogram(dat.norm(p=p_unest['p']),ax=ax)
    ax.set_ylim((0,5))
    ax.set_xlabel(r'$||x||_p$',fontsize=15)
    ax.set_ylabel(r'$\rho(||x||_p)$',fontsize=15)
    box_off(ax)

    ax = fig.add_subplot(236)
    p_est['rp'].histogram(dat.norm(p=p_est['p']),ax=ax)
    ax.set_ylim((0,5))
    ax.set_xlabel(r'$||x||_p$',fontsize=15)
    ax.set_ylabel(r'$\rho(||x||_p)$',fontsize=15)
    box_off(ax)

    fig.subplots_adjust(hspace=.3, wspace=.3)
    show()
#================= nonlinear transforms and log-determinants =====================
elif int(sys.argv[1]) == 3:
    from natter.Transforms import NonlinearTransformFactory 
    from natter.Distributions import ISA, ExponentialPower, ISA, Uniform
    from natter.Logging.LogTokens import Table
    # ----- loading and preprocessing data ----------
    print "Loading a simple Data module from an ascii file"
    FILENAME_TRAIN = 'hateren8x8_train_No1.dat.gz'

    dat = DataLoader.load(FILENAME_TRAIN)

    dat.center()
    dat.makeWhiteningVolumeConserving()

    FDCAC = LinearTransformFactory.DCAC(dat)
    FDC = FDCAC[0,:]
    FAC = FDCAC[1:,:]
    FwPCA = LinearTransformFactory.wPCA(FAC*dat)
    dat = FwPCA*FAC*dat

    # -- fitting distributions and creating transforms 
    n = dat.dim()
    p = LpSphericallySymmetric(n=n)
    pT = ISA(P=[Uniform() for _ in xrange(n)],n=n,S=[(i,) for i in xrange(n)])
    pISA_before = ISA(P=[Histogram() for _ in xrange(n)],n=n,S=[(i,) for i in xrange(n)])
    pISA_after = ISA(P=[Histogram() for _ in xrange(n)],n=n,S=[(i,) for i in xrange(n)])

    # -- copula without radial factorization before
    p.estimate(dat)
    pISA_before.estimate(dat,bins=5000)

    FHE = NonlinearTransformFactory.MarginalHistogramEqualization(pISA_before,pT)
    dat_no_rf = FHE*dat

    # -- copula with radial factorization before
    FRF = NonlinearTransformFactory.RadialFactorization(p)
    dat2 = FRF*dat

    pISA_after.estimate(dat2,bins=5000)
    FHE = NonlinearTransformFactory.MarginalHistogramEqualization(pISA_after,pT)
    dat_rf = FHE*dat2

    # -- plot the resulting marginals
    fig = figure()
    ax = fig.add_subplot(121)
    dat_no_rf[:2,:].plot(ax=ax,color='k')
    ax.set_title('without radial factorization',fontweight='bold',fontsize=20)
    ax.axis([0,1,0,1])
    ax.axis('off')

    ax = fig.add_subplot(122)
    dat_rf[:2,:].plot(ax=ax,color='k')
    ax.set_title('with radial factorization',fontweight='bold',fontsize=20)
    ax.axis([0,1,0,1])
    ax.axis('off')

    # -- correct average log-loss for transformation
    T = Table(['before rad. fac.', 'after rad. fac.'],
                     ['untransf. data','transf. data with log-det'])
    T['after rad. fac.','untransf. data'] = pISA_after.all(dat2)
    T['after rad. fac.','transf. data with log-det'] = pT.all(dat_rf) \
                        - mean(FHE.logDetJacobian(dat2))/log(2)/dat2.dim()
    T['before rad. fac.','transf. data with log-det'] = pT.all(dat_rf) \
                        - mean((FHE*FRF).logDetJacobian(dat))/log(2)/dat.dim()
    T['before rad. fac.','untransf. data'] = p.all(dat)
    print T
    show()


from __future__ import division
from FiniteMixtureDistribution import FiniteMixtureDistribution
from Gaussian import Gaussian
from numpy import zeros,ones,diag,eye,cov,log,sum,mean,sqrt,kron,reshape,dot,outer,exp
from numpy.linalg import cholesky,inv,solve,eigvalsh
from scipy import optimize
import sys
import pylab as pl
from natter.Auxiliary.Numerics import logsumexp

class FiniteMixtureOfGaussians(FiniteMixtureDistribution):
    """
    specialized class for a mixture of Gaussians.
    """

    def __init__(self,
                 numberOfMixtureComponents=5,
                 dim=4,
                 primary=None):
        """
        
        :arguments: 
           :param numberOfMixtureComponents: number of Gaussians
           :param dim:   dimensionality of each gaussian
           :param primary: which parameters to fit?
           
        :return:
            
        """
        if primary==None:
            primary = ['mu','sigma']
        baseDistribution = Gaussian({'n':dim})
        baseDistribution.primary = primary
        FiniteMixtureDistribution.__init__(self,numberOfMixtureComponents=numberOfMixtureComponents,
                                           baseDistribution=baseDistribution)
        
    def estimate(self,dat,verbose=False):
        """
        estimate the parameters of the mixture of Gaussians using a standard EM algorithm.
        """
        n,m = dat.size()
        K   = self.param['numberOfMixtureComponents']
        T = zeros((K,m))
        LP = zeros((K,m))
        diff = 100
        oldLP = 10000
        while diff>1e-09:
            for k in xrange(K):
                LP[k,:] = self.param['ps'][k].loglik(dat)  + log(self.param['alphas'][k])
            for k in xrange(K):
                T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))
            self.param['alphas'] = mean(T,axis=1)
            for k in xrange(K):
                TS = sum(T[k,:])
                if 'mu' in self.param['ps'][k].primary:
                    self.param['ps'][k].param['mu'] = sum(dat.X*T[k,:]/TS,axis=1)
                if 'sigma' in self.param['ps'][k].primary:
                    X = dat.X
                    n,m = X.shape
                    Y= (X - kron(reshape(self.param['ps'][k].param['mu'],(n,1)),ones((1,m))))
                    Y = Y*sqrt(T[k,:]/TS)
                    C = cov(Y)*(m-1) + eye(n)*1e-05 # add a ridge
                    # C = zeros((n,n))
                    # for l in xrange(m):
                    #     C = C + (T[k,l]/TS)*outer(Y[:,l],Y[:,l])
                    self.param['ps'][k].param['sigma'] = C
                    self.param['ps'][k].cholP[self.param['ps'][k].I] = solve(cholesky(self.param['ps'][k].param['sigma']),eye(n))[self.param['ps'][k].I]


            cALL=sum(-(T*LP).flatten())/(n*m)/log(2)
            diff = abs(oldLP-cALL)/abs(oldLP) # relative difference...
            print "\rrelative difference: " ,diff , "  current ALL: " , cALL ,"             ",
            sys.stdout.flush()
            oldLP = cALL

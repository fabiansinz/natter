from __future__ import division
from natter.Distributions import FiniteMixtureDistribution,  Gaussian, TruncatedGaussian, Gamma
from natter.Distributions import Gaussian
from numpy import log,pi,sum,array,ones,eye,sqrt,exp,mean,all
from numpy.linalg import norm,cholesky,inv
import unittest
from numpy.random import randn, rand
from scipy.optimize import check_grad, approx_fprime
import numpy
import pylab as pl
from mdp.utils import symrand
from natter.Auxiliary.Numerics import logsumexp
from matplotlib.pyplot import show
import numpy as np

class TestFiniteMixtureDistribution(unittest.TestCase):


    def setUp(self):
        self.n = 1
        self.K = 3
        self.nsamples = 10000
        self.a = 0.1
        self.b = 10
        alpha = rand(self.K)
        alpha = alpha/sum(alpha)
        P = [Gaussian(n=1,mu=3*randn(1),sigma=3*rand(1,1)) for k in xrange(self.K)]
        #P = [TruncatedGaussian(a=self.a,b=self.b,mu=3*randn(1),sigma=3*rand(1,1)+1) for k in xrange(self.K)]
        #P = [Gamma(n=1,u=3*rand(),s=3*rand()) for k in xrange(self.K)]
        self.mog = FiniteMixtureDistribution(P=P,alpha=alpha)
        self.mog.primary=['alpha','P']
        self.dat = self.mog.sample(self.nsamples)



    def test_primary2arrayConversion(self):
        p = self.mog.primary2array()
        mog2 = self.mog.copy()
        mog2.array2primary(p)
        p2 = mog2.primary2array()
        self.assertTrue(all(abs(p2-p)) < 1e-6,'Primary2array conversion does not leave parameters invariant')

    def test_loglik(self):
        nsamples = 1000000
        Gauss = Gaussian(n=1,mu=array([0]),sigma=array([[4]]))
        dat = Gauss.sample(nsamples)
        logWeights = self.mog.loglik(dat) - Gauss.loglik(dat)
        Z = logsumexp(logWeights)-log(nsamples)
        print "test_loglik: z: " ,exp(Z)
        self.assertTrue(abs(exp(Z)-1)<1e-01)


    def test_dldtheta(self):
        arr0 = self.mog.primary2array()
        def f(X):
            self.mog.array2primary(X)
            lv = self.mog.loglik(self.dat);
            slv = mean(lv)
            return slv
        def df(X):
            self.mog.array2primary(X)
            gv = self.mog.dldtheta(self.dat)
            sgv = mean(gv, axis=1);
            return sgv
        # arr0 = abs(randn(len(arr0)))+1
        err = check_grad(f,df,arr0)
        print "error in gradient: ", err
        self.assertTrue(err < 1e-01)

    def test_cdf_ppf(self):
        u = rand(10)
        u2 = self.mog.cdf(self.mog.ppf(u))
        self.assertTrue(all(abs(u-u2)< 1e-6),'ppf and cdf are not consistent!')

    def test_estimate(self):
        sigm1 = symrand(2)
        sigm1 = np.dot(sigm1,sigm1.T)
        sigm2 = symrand(2)
        sigm2 = np.dot(sigm2,sigm2.T)
        P = [Gaussian(n=2,mu=2*randn(2),sigma=sigm1), Gaussian(n=2,mu=2*randn(2),sigma=sigm2)]
        # P = [Gamma(n=1,u=3*rand(),s=3*rand()) for k in xrange(self.K)]
        # P = [TruncatedGaussian(a=0.1,b=10,mu=3*randn(1),sigma=3*rand(1,1)) for k in xrange(self.K)]
        alpha = rand(2)
        alpha = alpha/sum(alpha)
        mog = FiniteMixtureDistribution(P=P,alpha=alpha)
        mog.primary=['alpha','P']
        dat = mog.sample(50000)        
        arr0 = mog.primary2array()
        print mog        
        mog.array2primary(arr0 + np.random.rand(len(arr0))*1e-04)
        mog.estimate(dat,method='hybrid')
        err = np.sum(np.abs(arr0 - mog.primary2array()))
        print mog
        print  "error: ", err
        self.assert_(err < 1.0)


    # def test_ppf(self):
    #     u = rand(100)
    #     dat = self.mog.ppf(u,(0*u+self.a,0*u+self.b))
    #     # self.mog.histogram(dat)

    #     # dat2 = self.mog.sample(100000)
    #     # self.mog.histogram(dat2)
        
    #     # show()
    #     # raw_input()
        

 

##################################################################################################

if __name__=="__main__":
    unittest.main()



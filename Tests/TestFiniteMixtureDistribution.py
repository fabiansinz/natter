from __future__ import division
from natter.Distributions import FiniteMixtureDistribution,  Gaussian, TruncatedGaussian, Gamma
from natter.Distributions import Gaussian
from numpy import log,pi,sum,array,ones,eye,sqrt,exp,mean
from numpy.linalg import norm,cholesky,inv
import unittest
from numpy.random import randn, rand
from scipy.optimize import check_grad, approx_fprime
import numpy
import pylab as pl
from mdp.utils import symrand
from natter.Auxiliary.Numerics import logsumexp
from matplotlib.pyplot import show

class TestFiniteMixtureDistribution(unittest.TestCase):


    # def test_init(self):
    #     P = [Gaussian(n=1,mu=10*randn(1,1),sigma=5*rand(1,1)) for k in xrange(10)]
    #     p = FiniteMixtureDistribution(P=P)
    #     theta =  p.primary2array()
    #     theta *=2
    #     print p
    #     p.array2primary(theta)
    #     print p
    #     # dat =  p.sample(100000)
    #     # p.histogram(dat)
    #     # show()
    #     # raw_input()
    #     self.assertTrue(True)
    
    def setUp(self):
        self.n = 1
        self.K = 3
        self.nsamples = 10000
        self.a = 0.1
        self.b = 10
        alpha = rand(self.K)
        alpha = alpha/sum(alpha)
        #P = [Gaussian(n=1,mu=3*randn(1),sigma=3*rand(1,1)) for k in xrange(self.K)]
        P = [TruncatedGaussian(a=self.a,b=self.b,mu=3*randn(1),sigma=3*rand(1,1)+1) for k in xrange(self.K)]
        #P = [Gamma(n=1,u=3*rand(),s=3*rand()) for k in xrange(self.K)]
        self.mog = FiniteMixtureDistribution(P=P,alpha=alpha)
        self.mog.primary=['alpha','P']
        self.dat = self.mog.sample(self.nsamples)



    # def test_primary2arrayConversion(self):
    #     p = self.mog.primary2array()
    #     mog2 = self.mog.copy()
    #     mog2.array2primary(p)
    #     p2 = mog2.primary2array()
    #     self.assertTrue(all(abs(p2-p)) < 1e-6,'Primary2array conversion does not leave parameters invariant')

    # def test_loglik(self):
    #     nsamples = 1000000
    #     Gauss = Gaussian(n=1,mu=array([0]),sigma=array([[4]]))
    #     dat = Gauss.sample(nsamples)
    #     logWeights = self.mog.loglik(dat) - Gauss.loglik(dat)
    #     Z = logsumexp(logWeights)-log(nsamples)
    #     print "test_loglik: z: " ,exp(Z)
    #     self.assertTrue(abs(exp(Z)-1)<1e-01)


    # def test_dldtheta(self):
    #     arr0 = self.mog.primary2array()
    #     def f(X):
    #         self.mog.array2primary(X)
    #         lv = self.mog.loglik(self.dat);
    #         slv = mean(lv)
    #         return slv
    #     def df(X):
    #         self.mog.array2primary(X)
    #         gv = self.mog.dldtheta(self.dat)
    #         sgv = mean(gv, axis=1);
    #         return sgv
    #     # arr0 = abs(randn(len(arr0)))+1
    #     err = check_grad(f,df,arr0)
    #     print "error in gradient: ", err
    #     self.assertTrue(err < 1e-01)

    # def test_estimate(self):
    #     # P = [Gaussian(n=1,mu=2*randn(1),sigma=2*rand(1,1)) for k in xrange(self.K)]
    #     # P = [Gamma(n=1,u=3*rand(),s=3*rand()) for k in xrange(self.K)]
    #     P = [TruncatedGaussian(a=0.1,b=10,mu=3*randn(1),sigma=3*rand(1,1)) for k in xrange(self.K)]

    #     mog = FiniteMixtureDistribution(P=P)
    #     print mog
    #     mog.histogram(self.dat,cdf=True)
    #     show()
    #     mog.estimate(self.dat,method='hybrid')
    #     print mog
    #     mog.histogram(self.dat,cdf=True)
    #     print self.mog
    #     show()
    #     raw_input()


    def test_ppf(self):
        u = rand(100000)
        dat = self.mog.ppf(u,(0*u+self.a+1e-6,0*u+self.b-1e-6))
        self.mog.histogram(dat)

        dat2 = self.mog.sample(100000)
        self.mog.histogram(dat2)
        
        show()
        raw_input()
        

 

##################################################################################################

if __name__=="__main__":
    unittest.main()



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
        self.K = 4
        self.nsamples = 10000
        alpha = rand(self.K)
        alpha = alpha/sum(alpha)
        P = [Gamma(u=3*rand(),s=3*rand()) for k in xrange(self.K)]
        self.mog = FiniteMixtureDistribution(P=P,alpha=alpha)
        self.mog.primary=['alpha','P']
        self.dat = self.mog.sample(self.nsamples)



    def test_primary2arrayConversion(self):
        p = self.mog.primary2array()
        mog2 = self.mog.copy()
        mog2.array2primary(p)
        p2 = mog2.primary2array()
        self.assertTrue(all(abs(p2-p)) < 1e-6,'Primary2array conversion does not leave parameters invariant')
    # def test_loglik(self):
        
    #     nsamples = 1000000
    #     Gauss = Gamma(u=5,s=1)
    #     dat = Gauss.sample(nsamples)
    #     logWeights = self.mog.loglik(dat) - Gauss.loglik(dat)
    #     Z = logsumexp(logWeights)-log(nsamples)
    #     print "test_loglik: z: " ,exp(Z)
    #     self.assertTrue(abs(exp(Z)-1)<1e-01)


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

    def test_estimate(self):
        P = [Gamma(u=2*rand(),s=2*rand()) for k in xrange(self.K)]
        mog = FiniteMixtureDistribution(P=P)
        print mog
        mog.histogram(self.dat)
        show()
        mog.estimate(self.dat,method='hybrid')
        mog.histogram(self.dat)
        show()
        raw_input()
        print self.mog

    #     self.assertTrue(True)
#     def test_estimate(self):
#         """
#         test, if we can learn the same parameters with which we generated the data.
#         """
#         # arrStart = self.mixture.primary2array()
#         # self.mixture.alphas = array([0.8,0.2])
#         # self.mixture.etas = array([log(1/self.mixture.alphas[0] -1)])
#         # self.mixture.estimate(self.data,verbose=True)
#         # arrEnd   = self.mixture.primary2array()
#         # print "ground truth: " , arrStart
#         # print "ended with : ", arrEnd
#         # ALL = sum(self.mixture.loglik(self.data))
#         # print "Difference in ALL : " , abs(ALL - self.ALL)


#         # arrStart = self.GaussMixture.primary2array()
#         # self.GaussMixture.alphas = array([0.8,0.2])
#         # self.GaussMixture.etas = array([log(1/self.GaussMixture.alphas[0] -1)])
#         # self.GaussMixture.estimate(self.data,verbose=True)
#         # arrEnd   = self.GaussMixture.primary2array()
#         # print "ground truth: " , arrStart
#         # print "ended with : ", arrEnd
#         # ALL = sum(self.GaussMixture.loglik(self.data))
#         # print "Difference in ALL : " , abs(ALL - self.ALL)
#         pass
        


 

##################################################################################################

if __name__=="__main__":
    unittest.main()



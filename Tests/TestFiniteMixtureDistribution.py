from __future__ import division
from natter.Distributions import FiniteMixtureDistribution,  Gaussian, TruncatedGaussian
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
        K = 2
        self.nsamples = 5000
        P = [TruncatedGaussian(a=0,b=10,n=self.n,mu=2*randn(1,1),sigma=2*rand(1,1)) for k in xrange(K)]
        self.mog = FiniteMixtureDistribution(P=P)
        self.dat = self.mog.sample(self.nsamples)


    #     dim = 2
    #     self.dim=dim
    #     self.baseDistribution = Gaussian({'n':dim})
    #     self.baseDistribution.primary = ['sigma']
    #     self.numberOfMixtureComponents = 2;
    #     self.nsamples = 5000
    #     self.mixture = FiniteMixtureDistribution(numberOfMixtureComponents=self.numberOfMixtureComponents,
    #                                              baseDistribution=self.baseDistribution)
    #     C1 =  eye(dim)*5 + ones((dim,dim)) + symrand(dim)
    #     C2 =  eye(dim)*5  -ones((dim,dim))+ symrand(dim)
    #     self.mixture.param['ps'][0].param['sigma'] =C1
    #     self.mixture.param['ps'][0].cholP  = cholesky(inv(C1))
    #     self.mixture.param['ps'][1].param['sigma'] =C2
    #     self.mixture.param['ps'][1].cholP  = cholesky(inv(C2))
    #     self.mixture.alphas = array([0.6,0.4])
    #     self.mixture.etas = array([log(1/self.mixture.alphas[0] -1)])
    #     self.GaussMixture = FiniteMixtureOfGaussians(numberOfMixtureComponents=2,dim=dim,primary=['sigma'])
    #     C1 =  eye(dim)*5 + ones((dim,dim))
    #     C2 =  eye(dim)*5  -ones((dim,dim))
    #     self.GaussMixture.param['ps'][0].param['sigma'] =C1
    #     self.GaussMixture.param['ps'][0].cholP  = cholesky(inv(C1))
    #     self.GaussMixture.param['ps'][1].param['sigma'] =C2
    #     self.GaussMixture.param['ps'][1].cholP  = cholesky(inv(C2))
    #     self.GaussMixture.alphas = array([0.6,0.4])
    #     self.GaussMixture.etas = array([log(1/self.GaussMixture.alphas[0] -1)])
    #     self.data = self.mixture.sample(self.nsamples)
    #     self.ALL = sum(self.mixture.loglik(self.data))
        
#     def test_init(self):
#         pass

#     def test_sample(self):
#         nsamples = 1000000
#         Gauss = Gaussian({'n':self.dim})
#         data = self.mixture.sample(nsamples)
#         logWeights = Gauss.loglik(data) -self.mixture.loglik(data)
#         Z = logsumexp(logWeights)-log(nsamples)
#         print "test_sample: z: " ,exp(Z)
#         self.assertTrue(abs(exp(Z)-1)<1e-01)

    def test_loglik(self):
        
        nsamples = 1000000
        Gauss = TruncatedGaussian(n=self.n,a=0,b=10)
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

        arr0 = abs(randn(len(arr0)))+1
        err = check_grad(f,df,arr0)
        print "error in gradient: ", err
        self.assertTrue(err < 1e-01)

        
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



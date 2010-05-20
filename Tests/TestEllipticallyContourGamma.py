from __future__ import division
from natter.Distributions import EllipticallyContourGamma,Gaussian
from numpy import log,pi,sum,ones,sqrt,logspace,exp,eye
from natter.Transforms import LinearTransform
from numpy.random import randn
from numpy.linalg import norm
import unittest
from scipy.optimize import check_grad, approx_fprime
import numpy
from mdp.utils import random_rot
from natter.Auxiliary.Numerics import logsumexp

class TestEllipticallyContourGamma(unittest.TestCase):
    def setUp(self):
        self.n =2
        self.W = LinearTransform(eye(self.n))
        self.ECG = EllipticallyContourGamma({'n': self.n,
                                             'W': self.W})
        self.Gaussian = Gaussian({'n':self.n})
        self.data = self.Gaussian.sample(1000)
        
    def test_init(self):
        pass


    def test_array2primary(self):
        abefore = self.ECG.primary2array();
        #arrr = randn(len(abefore))
        #        abefore=arrr
        #self.ECG.array2primary(arrr)
        #aafter  = self.ECG.primary2array()
        #diff = norm(abefore-aafter)
        #self.assertTrue(diff <=1e-05)

    def test_loglik(self):
        nsamples=100000
        dataImportance = self.Gaussian.sample(nsamples)
        logweights = self.ECG.loglik(dataImportance)-self.Gaussian.loglik(dataImportance)
        Z = logsumexp(logweights)-log(nsamples)
        print "sampled partition function:", exp(Z)
        
    

    def test_dldtheta(self):

        self.ECG.primary = ['q']
        def f(X):
            self.ECG.array2primary(X)
            lv = self.ECG.loglik(self.data);
            slv = sum(lv)
            return slv
        def df(X):
            self.ECG.array2primary(X)
            gv = self.ECG.dldtheta(self.data)
            sgv = sum(gv, axis=1);
            return sgv
        theta0 = self.ECG.primary2array()
        theta0 = abs(randn(len(theta0)))+1
        err = check_grad(f,df,theta0)
        print "error in gradient: ", err
        self.ECG.primary = ['W']
        def f2(X):
            self.ECG.array2primary(X)
            lv = self.ECG.loglik(self.data);
            slv = sum(lv)
            return slv
        def df2(X):
            self.ECG.array2primary(X)
            gv = self.ECG.dldtheta(self.data)
            sgv = sum(gv, axis=1);
            return sgv
        theta0 = self.ECG.primary2array()
        theta0 = abs(randn(len(theta0)))+1
        err = check_grad(f2,df2,theta0)
        print "error in gradient: ", err
        self.assertTrue(err < 1e-02)
        

##################################################################################################

if __name__=="__main__":
    unittest.main()



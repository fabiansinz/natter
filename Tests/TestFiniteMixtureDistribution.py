from __future__ import division
from natter.Distributions import FiniteMixtureDistribution, FiniteMixtureOfGaussians
from natter.Distributions import Gaussian
from numpy import log,pi,sum,array,ones,eye,sqrt
from numpy.linalg import norm,cholesky,inv
import unittest
from numpy.random import randn
from scipy.optimize import check_grad, approx_fprime
import numpy
import pylab as pl


class TestFiniteMixtureDistribution(unittest.TestCase):
    def setUp(self):
        dim = 2
        self.baseDistribution = Gaussian({'n':dim})
        self.baseDistribution.primary = ['sigma']
        self.numberOfMixtureComponents = 2;
        self.nsamples = 5000
        self.mixture = FiniteMixtureDistribution(numberOfMixtureComponents=self.numberOfMixtureComponents,
                                                 baseDistribution=self.baseDistribution)
        C1 =  eye(dim)*5 + ones((dim,dim))
        C2 =  eye(dim)*5  -ones((dim,dim))
        self.mixture.ps[0].param['sigma'] =C1
        self.mixture.ps[0].cholP  = cholesky(inv(C1))
        self.mixture.ps[1].param['sigma'] =C2
        self.mixture.ps[1].cholP  = cholesky(inv(C2))
        self.mixture.alphas = array([0.6,0.4])

        self.GaussMixture = FiniteMixtureOfGaussians(numberOfMixtureComponents=2,dim=dim,primary=['sigma'])
        C1 =  eye(dim)*5 + ones((dim,dim))
        C2 =  eye(dim)*5  -ones((dim,dim))
        self.GaussMixture.ps[0].param['sigma'] =C1
        self.GaussMixture.ps[0].cholP  = cholesky(inv(C1))
        self.GaussMixture.ps[1].param['sigma'] =C2
        self.GaussMixture.ps[1].cholP  = cholesky(inv(C2))
        self.GaussMixture.alphas = array([0.6,0.4])

        self.data = self.mixture.sample(self.nsamples)
        self.ALL = sum(self.mixture.loglik(self.data))
    def test_init(self):
        pass

    def test_dldtheta(self):
        arr0 = self.mixture.primary2array()
        def f(X):
            self.mixture.array2primary(X)
            lv = self.mixture.loglik(self.data);
            slv = sum(lv)
            return slv
        def df(X):
            self.mixture.array2primary(X)
            gv = self.mixture.dldtheta(self.data)
            sgv = sum(gv, axis=1);
            return sgv

        arr0 = abs(randn(len(arr0)))+1
        err = check_grad(f,df,arr0)
        print "error in gradient: ", err
        self.assertTrue(err < 1e-02)

        
    def test_estimate(self):
        """
        test, if we can learn the same parameters with which we generated the data.
        """

        arrStart = self.mixture.primary2array()
        self.mixture.alphas = array([0.8,0.2])
        self.mixture.estimate(self.data,verbose=True)
        arrEnd   = self.mixture.primary2array()
        print "ground truth: " , arrStart
        print "ended with : ", arrEnd
        ALL = sum(self.mixture.loglik(self.data))
        print "Difference in ALL : " , abs(ALL - self.ALL)


        arrStart = self.GaussMixture.primary2array()
        self.GaussMixture.alphas = array([0.8,0.2])
        self.GaussMixture.estimate(self.data,verbose=True)
        arrEnd   = self.GaussMixture.primary2array()
        print "ground truth: " , arrStart
        print "ended with : ", arrEnd
        ALL = sum(self.GaussMixture.loglik(self.data))
        print "Difference in ALL : " , abs(ALL - self.ALL)

        


 

##################################################################################################

if __name__=="__main__":
    unittest.main()



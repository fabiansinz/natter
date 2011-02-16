from __future__ import division
from natter.Distributions import Gaussian
from numpy import log,pi,sum,ones,sqrt,logspace
from numpy.random import randn
from numpy.linalg import norm
import unittest
from scipy.optimize import check_grad, approx_fprime
import numpy



class TestGaussian(unittest.TestCase):
    def setUp(self):
        self.Gauss = Gaussian({'n':1})
        self.entropy = 0.5*(1+log(2*pi))
        self.Gauss2D = Gaussian({'n':2})
        self.Gauss2D.primary=['sigma']
    def test_init(self):
        pass

    def test_sample(self):
        d = self.Gauss.sample(10)
        pass

    def test_loglik(self):
        nsamples=10000
        dat=self.Gauss.sample(nsamples)
        lv = self.Gauss.loglik(dat)
        hs = sum(-lv)/nsamples          # sampled entropy
        self.assertTrue( abs(hs-self.entropy)<= 1e-02 )


    def test_array2primary(self):
        arr = self.Gauss.primary2array();
        pbefore = self.Gauss.param.copy()
        arrr = randn(len(arr))
        self.Gauss.array2primary(arrr)
        self.Gauss.array2primary(arr);
        pafter  = self.Gauss.param.copy()
        diff = 0.0;
        for key in pbefore.keys():
            diff += norm(pbefore[key] - pafter[key])
        self.assertTrue(diff <=1e-05)

    def test_dldtheta(self):
        d = self.Gauss2D.sample(1)
        def f(X):
            self.Gauss2D.array2primary(X)
            lv = self.Gauss2D.loglik(d);
            slv = sum(lv)
            return slv
        def df(X):
            self.Gauss2D.array2primary(X)
            gv = self.Gauss2D.dldtheta(d)
            sgv = sum(gv, axis=1);
            return sgv
        theta0 = self.Gauss2D.primary2array()
        theta0 = abs(randn(len(theta0)))+1
        err = check_grad(f,df,theta0)
        print "error in gradient: ", err
        self.assertTrue(err < 1e-02)
        

    def test_estimate(self):
        data = self.Gauss2D.sample(500)
        self.Gauss2D.primary= ['sigma']
        thetaOrig = self.Gauss2D.primary2array()
        theta0 = abs(randn(len(thetaOrig)))
        self.Gauss2D.array2primary(theta0)
        self.Gauss2D.estimate(data,method="gradient")
        thetaOpt = self.Gauss2D.primary2array()
        err = thetaOpt-thetaOrig
        print "Error in thetas with gradient: " , norm(err)

        theta0 = abs(randn(len(thetaOrig)))
        self.Gauss2D.array2primary(theta0)
        self.Gauss2D.estimate(data,method="analytic")
        thetaOpt = self.Gauss2D.primary2array()
        err = thetaOpt-thetaOrig
        print "Error in thetas maxL: " , norm(err)
        self.assertTrue((norm(err)<1e-01))

##################################################################################################

if __name__=="__main__":
    unittest.main()



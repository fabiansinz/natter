from __future__ import division
from natter.Distributions import NakaRushton,Gamma
from numpy import log,pi,sum,ones,sqrt,logspace,exp,eye,cov,mean,linspace
from natter.DataModule import Data
from numpy.random import randn, rand
import unittest
from scipy.optimize import check_grad, approx_fprime
from natter.Auxiliary.Numerics import logsumexp\
 

class TestNakaRushton(unittest.TestCase):
    def setUp(self):
        self.n = 5
        self.kappa = 10*rand()
        self.s = 1.0
        self.sigma = 10*rand()
        self.p = 2.0
        self.P = NakaRushton(n = self.n,kappa=self.kappa,s =self.s,p =self.p,sigma=self.sigma)
        self.Gamma = Gamma(s=2.0,u=2.0)
        self.data = self.Gamma.sample(10000)
        self.f = lambda dat: (self.kappa*dat.X/sqrt(self.sigma**2.0 + abs(dat.X)**self.p)).reshape((1,dat.numex()))

    def test_loglik(self):
        nsamples=50000
        dataImportance = self.Gamma.sample(nsamples)
        logweights = self.P.loglik(dataImportance)-self.Gamma.loglik(dataImportance)
        Z = logsumexp(logweights)-log(nsamples)
        self.assertTrue(abs(exp(Z)-1)<1e-01)
                        

    def test_Partition(self):
        t = linspace(1e-12,100,50000)
        dt = t[1]-t[0]
        Z = sum(self.P.pdf(Data(t)))*dt
        self.assertTrue(abs(Z-1.0) < 5.0*1e-2)
        
    
    def test_cov(self):
        nsamples = 50000
        X = randn(self.n,nsamples)
        X = X/sqrt(sum(X**2,axis=0).reshape((1,nsamples)))
        r = self.P.sample(nsamples)
        r2 = self.f(r)
        X = X*r2
        self.assertTrue(mean(abs(eye(self.n) - cov(X)).flatten())< 5.0*1e-2)
        
    


    def test_estimate(self):
        P =  NakaRushton(n = self.n,kappa=self.kappa,s =self.s,p =self.p,sigma=self.sigma)
        dat = self.P.sample(20000)
        P.estimate(dat)
        self.assertTrue(abs(self.sigma - P['sigma'])<0.1)

    def test_dldtheta(self):
        self.P.primary = ['sigma']
        def f(X):
            self.P.array2primary(X)
            lv = self.P.loglik(self.data);
            slv = sum(lv)
            return slv
        def df(X):
            self.P.array2primary(X)
            gv = self.P.dldtheta(self.data)
            sgv = sum(gv, axis=1);
            return sgv
        theta0 = self.P.primary2array()
        theta0 = abs(randn(len(theta0)))+1
        err = check_grad(f,df,theta0)
        print "error in gradient: ", err
        self.assertTrue(err < 1e-02)
        

##################################################################################################

if __name__=="__main__":
    unittest.main()



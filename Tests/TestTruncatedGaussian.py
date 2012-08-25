from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from natter.Auxiliary import Errors
from scipy import io
from natter import Auxiliary
import sys
from scipy  import optimize
from natter.Auxiliary.Numerics import logsumexp
from numpy.random import rand
from matplotlib.pyplot import show
class TestTruncatedGaussian(unittest.TestCase):


    Tol = 1e-7
    TolParam = 1e-1

    def setUp(self):
        self.a = 1.0*rand()
        self.b = self.a+ 10.0*rand()
        self.mu = (self.a+10*rand())/2.
        self.s = 2.0*rand()+1.0
        self.p = Distributions.TruncatedGaussian({'a':self.a,'b':self.b,'mu':self.mu,'sigma':self.s})

    
    def test_loglik(self):
        p1 = self.p
        p2 = self.p.copy()
        p2['mu'] *= 1.1
        
        nsamples = 1000000
        data = p2.sample(nsamples)
        logZ = logsumexp(p1.loglik(data) -p2.loglik(data) - np.log(nsamples))
        print np.exp(logZ)
        print "Estimated partition function: ", np.exp(logZ)
        self.assertTrue(np.abs(np.exp(logZ)-1.0) < 0.1*self.TolParam,'Difference in estimated partition function (1.0) greater than' + str(0.1*self.TolParam))


    def test_cdf(self):
        print "Testing consistency of cdf and ppf"
        sys.stdout.flush()
            
        p = self.p
        u = rand(10)
        u2 = p.cdf(p.ppf(u))
        self.assertFalse(np.sum(np.abs(u-u2)) > self.TolParam,'Difference u - cdf(ppf(u)) greater than %.4g' %( self.Tol,))



    def test_dldtheta(self):
        OK =np.zeros(5)
        for i in range(5):
            self.a = 1.0*rand()
            self.b = self.a+ 10.0*rand()
            self.mu = (self.a+10*rand())/2.
            self.s = 2.0*rand()+1.0
            self.p = Distributions.TruncatedGaussian({'a':self.a,'b':self.b,'mu':self.mu,'sigma':self.s})
            p = self.p.copy()   
            p.primary=['mu','sigma']
            dat = p.sample(100)

            def f(arr):
                p.array2primary(arr)
                return np.sum(p.loglik(dat))
            def df(arr):
                p.array2primary(arr)
                return np.sum(p.dldtheta(dat),axis=1)

            arr0= p.primary2array()
            arr0 = abs(np.random.randn(len(arr0)))
            err  = optimize.check_grad(f,df,arr0)
            if err<1e-02:
                OK[i]=1
        M = np.max(OK)
        self.assertTrue(M>0.5,'Gradient error %.4g is greater than %.4g' % (err,1e-02))


        
        

if __name__=="__main__":
    
    unittest.main()

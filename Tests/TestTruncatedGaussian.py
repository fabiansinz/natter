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
    
    def test_loglik(self):
        p1 = Distributions.TruncatedGaussian({'a':1.0,'b':3.0,'mu':1.0,'sigma':2.0})
        p2 = Distributions.TruncatedGaussian({'a':1.0,'b':3.0,'mu':.5,'sigma':1.5})
        nsamples = 1000000
        data = p2.sample(nsamples)
        logZ = logsumexp(p1.loglik(data) -p2.loglik(data) - np.log(nsamples))
        print "Estimated partition function: ", np.exp(logZ)
        self.assertTrue(np.abs(np.exp(logZ)-1.0) < 0.1*self.TolParam,'Difference in estimated partition function (1.0) greater than' + str(0.1*self.TolParam))


    def test_cdf(self):
        print "Testing consistency of cdf and ppf"
        sys.stdout.flush()
        a = 1.0*rand()
        b = a
        while b <= a:
            b = 10.0*rand()
        mu = 10*rand()
        s = 2.0*rand()+1.0
            
        p = Distributions.TruncatedGaussian({'a':a,'b':b,'mu':mu,'sigma':s})   
        u = rand(10)
        u2 = p.cdf(p.ppf(u))

        self.assertFalse(np.sum(np.abs(u-u2)) > self.TolParam,'Difference u - cdf(ppf(u)) greater than %.4g' %( self.Tol,))



    def test_dldtheta(self):
        a = 1.0*rand()
        b = a
        while b <= a:
            b = 10.0*rand()
        mu = 10*rand()
        s = 2.0*rand()+1.0
            
        p = Distributions.TruncatedGaussian({'a':a,'b':b,'mu':mu,'sigma':s})
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
        print "Error in gradient: ",err
        self.assertTrue(err<1e-02,'Gradient error %.4g is greater than %.4g' % (err,1e-02))


        
        

if __name__=="__main__":
    
    unittest.main()

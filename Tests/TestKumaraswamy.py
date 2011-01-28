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

class TestKumaraswamy(unittest.TestCase):


    Tol = 1e-7
    TolParam = 1e-1
    
    def test_loglik(self):
        p1 = Distributions.Kumaraswamy({'a':2.0,'b':3.0})
        p2 = Distributions.Kumaraswamy({'a':1.0,'b':1.0})
        nsamples = 1000000
        data = p2.sample(nsamples)
        logZ = logsumexp(p1.loglik(data) -p2.loglik(data) - np.log(nsamples))
        print "Estimated partition function: ", np.exp(logZ)
        self.assertTrue(np.abs(np.exp(logZ)-1.0) < 0.1*self.TolParam,'Difference in estimated partition function (1.0) greater than' + str(0.1*self.TolParam))
        
        


    def test_estimate(self):
        print "Testing parameter estimation of Kumaraswamy distribution ..."
        sys.stdout.flush()
        myu = 10*rand()
        mys = 10*rand()
        myB = 10*rand()
        p = Distributions.Kumaraswamy({'a':myu ,'b':mys,'B':myB})
        dat = p.sample(50000)
        p = Distributions.Kumaraswamy(B=myB)
        p.estimate(dat)
        self.assertFalse(np.abs(p.param['a'] - myu) > self.TolParam,'Difference in Shape parameter for Kumaraswamy distribution greater than ' + str(self.TolParam))
        self.assertFalse(np.abs(p.param['b'] - mys) > self.TolParam,'Difference in Scale parameter for Kumaraswamy distribution greater than ' + str(self.TolParam))

    def test_cdf(self):
        print "Testing consistency of cdf and ppf"
        sys.stdout.flush()
        myu = 10*rand()
        mys = 10*rand()
        myB = 10*rand()
        p = Distributions.Kumaraswamy({'a':myu ,'b':mys,'B':myB})
        u = rand(10)
        u2 = p.cdf(p.ppf(u))

        self.assertFalse(sum(np.abs(u-u2)) > self.TolParam,'Difference u - cdf(ppf(u)) greater than %.4g' %( self.Tol,))



    def test_dldtheta(self):
        p = Distributions.Kumaraswamy({'a':5.0*rand(),'b':5.0*rand(),'B':10.0*rand()})
        p.primary=['a','b']
        dat = p.sample(1000)
        def f(arr):
            p.array2primary(arr)
            return np.sum(p.loglik(dat))
        def df(arr):
            p.array2primary(arr)
            return np.sum(p.dldtheta(dat),axis=1)
        arr0= p.primary2array()
        arr0 = abs(np.random.randn(len(arr0)))
        err  = optimize.check_grad(f,df,arr0)
        print "Error in graident: ",err
        self.assertTrue(err<1e-02,'Gradient error %.4g is greater than %.4g' % (err,1e-02))


        
        

if __name__=="__main__":
    
    unittest.main()

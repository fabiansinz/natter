from __future__ import division
from natter.Distributions import Gaussian, Boltzmann
from numpy import log,pi,sum,ones,sqrt,logspace,exp,zeros
from numpy.random import randn
from numpy.linalg import norm
import unittest
from scipy.optimize import check_grad, approx_fprime
import numpy
from natter.DataModule import Data
from natter.Auxiliary.Numerics import logsumexp


class TestBoltzmann(unittest.TestCase):
    def setUp(self):
        n=2
        L = zeros((n,n))
        L[n-1,0]=1
        self.model = Boltzmann({'n':2,'L':L})
        

    def test_init(self):
        pass

    def test_loglik(self):
        nsamples = 100000
        self.model.estimatePartitionFunction()
        dataImportance = Data(1.*randn(self.model.param['n'],nsamples)>0)         # uniform binomial
        logweights = self.model.loglik(dataImportance) + self.model.param['n']*log(2)
        Z = logsumexp(logweights)-log(nsamples)
        print "estimated Z . ", exp(Z)
        self.assertTrue(abs(exp(Z)-1)<1e-01)


    def test_sample(self):
        nsamples=1000
        data = self.model.sample(nsamples,sampleOpts={'method':'Gibbs',
                                                      'burnIn':1000,
                                                      'indepGap':10})
        self.model.estimatePartitionFunction()
        logweights = -self.model.param['n']*log(2)-self.model.loglik(data)
        Z = logsumexp(logweights)-log(nsamples)
        print "estimated Z . ", exp(Z)        
        self.assertTrue(abs(exp(Z)-1)<1e-01)        

    def test_estimatePartitionFunction(self):

        self.model.estimatePartitionFunction({'method':'brute'})
        logZ1 = self.model.param['logZ']
        print "logZ1: ",logZ1
        self.model.estimatePartitionFunction({'method':'importance'})
        logZ2 = self.model.param['logZ']
        print "logZ2: ",logZ2
        self.assertTrue(abs(logZ1-logZ2)<1e-01)
                                                      
    



##################################################################################################

if __name__=="__main__":
    unittest.main()

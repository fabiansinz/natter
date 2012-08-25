from __future__ import division
import unittest
from natter.Auxiliary import Entropy
from sys import stdout
from numpy import log, pi, e, max, abs
from numpy.random import rand, randn, randint
from natter.DataModule import Data

class TestEntropy(unittest.TestCase):

    tol = 5.0*1e-2

    def test_marginalEntropyEstimators(self):
        print "Testing marginal entropy estimation ..."; stdout.flush()
        s = 10.0*rand(2,1)
        x = randn(2,10000)*s
        h = .5*log(2.0*pi*e*s**2)
        dat = Data(x)
        for method in ['MLE','JK','CAE','MM']:
            h2 = Entropy.marginalEntropy(dat,method)
            self.assertTrue(max(abs(h-h2)) < self.tol, 'Entropy estimates for ' + method + 'differ by more than ' + str(self.tol))
        

    def test_LpEntropy(self):
        print "Testing Lp-Entropy estimator"
        n = randint(5)+1
        s = 10.0*rand()
        x = randn(n,20000)*s
        h = n*.5*log(2.0*pi*e*s**2)
        dat = Data(x)
        h2 = Entropy.LpEntropy(dat)
        self.assertTrue(abs(h-h2) < self.tol, 'Entropy estimates for LpEntropy differ by more than ' + str(self.tol))
        


if __name__=="__main__":
    
    unittest.main()

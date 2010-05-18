from __future__ import division
from natter.Distributions import FiniteMixtureDistribution
from natter.Distributions import Gaussian
from numpy import log,pi,sum,array
from numpy.linalg import norm
import unittest
from scipy.optimize import check_grad, approx_fprime
import numpy



class TestFiniteMixtureDistribution(unittest.TestCase):
    def setUp(self):
        self.baseDistribution = Gaussian({'n':1})
        self.baseDistribution.primary = ['mu']
        self.numberOfMixtureComponents = 2;
        self.nsamples = 2000
        self.mixture = FiniteMixtureDistribution(numberOfMixtureComponents=self.numberOfMixtureComponents,
                                                 baseDistribution=self.baseDistribution)
        self.mixture.ps[1].param['mu'] = 15.0
        self.data = self.mixture.sample(self.nsamples)

    def test_init(self):
        pass

    def test_estimate(self):
        """
        test, if we can learn the same parameters with which we generated the data.
        """
        arrStart = self.mixture.primary2array()
        # self.mixture.ps[0].param['mu'] =0.5
        # self.mixture.ps[1].param['mu'] = 1.0
        # self.mixture.alphas = array([0.8,0.2])
        self.mixture.estimate(self.data,verbose=True)
        arrEnd   = self.mixture.primary2array()
        print "ground truth: " , arrStart
        print "ended with : ", arrEnd
        print "Error in parameters: " , norm(arrStart-arrEnd)


 

##################################################################################################

if __name__=="__main__":
    unittest.main()



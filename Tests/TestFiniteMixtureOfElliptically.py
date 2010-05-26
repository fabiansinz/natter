from __future__ import division
from natter.Distributions import EllipticallyContourGamma,Gaussian,FiniteMixtureOfGaussians,FiniteMixtureDistribution,FiniteMixtureOfEllipticallyGamma
from numpy import log,pi,sum,ones,sqrt,logspace,exp,eye,eye,ones
from natter.Transforms import LinearTransform
from numpy.random import randn
from numpy.linalg import norm,cholesky,inv
import unittest
from scipy.optimize import check_grad, approx_fprime
import numpy
from mdp.utils import random_rot
from natter.Auxiliary.Numerics import logsumexp

class TestEllipticallyContourGamma(unittest.TestCase):
    def setUp(self):
        self.nc=2
        self.n =2
        self.W = LinearTransform(eye(self.n))
        self.ECG = EllipticallyContourGamma({'n': self.n,
                                             'W': self.W})
        self.Gaussian = Gaussian({'n':self.n})
        self.MOG = FiniteMixtureOfGaussians(numberOfMixtureComponents=self.nc,
                                            dim=self.n,
                                            primary=['sigma'])
        C1 =  eye(self.n)*5 + ones((self.n,self.n))
        C2 =  eye(self.n)*5  -ones((self.n,self.n))
        self.MOG.ps[0].param['sigma'] =C1
        self.MOG.ps[0].cholP  = cholesky(inv(C1))
        self.MOG.ps[1].param['sigma'] =C2
        self.MOG.ps[1].cholP  = cholesky(inv(C2))
        self.data = self.MOG.sample(10000)

        self.MECG = FiniteMixtureDistribution(baseDistribution=self.ECG,
                                              numberOfMixtureComponents=self.nc)
        self.M2ECG = FiniteMixtureOfEllipticallyGamma({'n':self.n,
                                                       'NC':self.nc})
        
    def test_init(self):
        pass

    def test_estimate(self):
        self.M2ECG.estimate(self.data)
        print "Difference in ALL (MOG -M2ECG): ", abs(self.MOG.all(self.data)-self.M2ECG.all(self.data))
        self.M2ECG.estimate(self.data,method='gradient')
        print "Difference in ALL for gradient method (MOG -M2ECG): ", abs(self.MOG.all(self.data)-self.M2ECG.all(self.data))
        self.MECG.estimate(self.data)
        print "Difference in ALL (MOG -MECG): ", abs(self.MOG.all(self.data)-self.MECG.all(self.data))
        print self.MECG
##################################################################################################

if __name__=="__main__":
    unittest.main()



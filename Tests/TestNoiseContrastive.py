from natter.Auxiliary.Estimation import noiseContrastive
from natter.Auxiliary import profileFunction
from natter.Distributions import UnnormalizedGaussian
from numpy import zeros,eye,ones
from numpy.random import randn
from numpy.linalg import norm
import unittest

class TestNoiseContrastive(unittest.TestCase):
    def test_estimate(self):
        nSamples = 10000;
        modelParam = {'n': 2,'mu':zeros(2),'sigma': eye(2)}
        modelDist = UnnormalizedGaussian(param=modelParam)
        trueArr = modelDist.primary2array();
        data = modelDist.sample(nSamples)
        mustart = randn(2)
        modelDist.setParam('mu',mustart)
        def f():
            noiseContrastive(modelDist,data,verbosity=1)
        profileFunction(f)
        estimateArr = modelDist.primary2array()
        self.assertTrue(norm(trueArr - estimateArr)<= 1e-01)
        
        
        




##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

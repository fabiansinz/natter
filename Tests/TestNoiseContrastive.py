from natter.Auxiliary.Estimation import noiseContrastive
from natter.Distributions import Gaussian
from numpy import zeros,eye,ones
from numpy.random import randn
from numpy.linalg import norm
import unittest

class TestNoiseContrastive(unittest.TestCase):
    def test_estimate(self):
        nSamples = 1000;
        modelParam = {'n': 2,'mu':ones(2),'sigma': eye(2)}
        modelDist = Gaussian(param=modelParam)
        trueArr = modelDist.primary2array();
        data = modelDist.sample(nSamples)
        mustart = randn(2)
        modelDist.setParam('mu',mustart)
        noiseContrastive(modelDist,data,verbosity=1)
        estimateArr = modelDist.primary2array()
        self.assertTrue(norm(trueArr - estimateArr)<= 1e-01)
        
        
        




##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

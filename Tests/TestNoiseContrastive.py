from natter.Auxiliary.Estimation import noiseContrastive
from natter.Auxiliary import profileFunction
from natter.Distributions import UnnormalizedGaussian
from numpy import zeros,eye,ones
#from numpy.random import randn
from numpy.linalg import norm
import unittest

class TestNoiseContrastive(unittest.TestCase):
    def test_estimate(self):
        nSamples = 3000;
        modelParam = {'n': 2,'mu':zeros(2),'sigma': eye(2)}
        modelDist = UnnormalizedGaussian(param=modelParam)
        trueArr = modelDist.primary2array();
        mustart = ones(2)*5
        covstart = eye(2)
        modelDist.setParam('mu',mustart)
        modelDist.setParam('sigma',covstart)
        modelDist.primary = ['Z']
        data = modelDist.sample(nSamples)
        noiseContrastive(modelDist,data,verbosity=1)
        estimateArr = modelDist.primary2array()
        print "error: ", (norm(trueArr - estimateArr))
        pass
        
        
        




##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    


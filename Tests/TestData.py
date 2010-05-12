import unittest
from numpy.random import randn
from numpy import dot, abs
from numpy.linalg import det, cholesky
from natter.Transforms import LinearTransformFactory
from natter.DataModule import Data
class TestData(unittest.TestCase):
    
    def test_makeWhiVolCons(self):
        print "Testing making whitening volume conserving ...",
        C = randn(5,5)
        C = dot(C,C.T)
        
        dat = Data(dot(cholesky(C),randn(5,10000)))
        dat.makeWhiteningVolumeConserving()
        F0 = LinearTransformFactory.DCnonDC(dat)
        F0 = F0[1:,:]
        F = LinearTransformFactory.SYM(F0*dat)
        self.assertTrue(abs(det(F.W) - 1.0) < 1e-5,'Determinant is not equal to one!')
        







##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

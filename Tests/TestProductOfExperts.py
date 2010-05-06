import unittest
import Distribution
import Data
import sys
from numpy.random import randn, rand
from numpy import max, abs, size, array

class TestProductOfExperts(unittest.TestCase):

    Tol = 1e-5
    
    def test_scorefct(self):
        print "Testing score function for Product of Experts ..."
        sys.stdout.flush()
        
        dat = Data.Data(randn(5,100))
        p = Distribution.ProductOfExperts({'n':5})
        
        alpha0 = rand(p.param['N'])
    
        W0 = array(p.param['W'].W.copy())

    
        h = 1e-8
        dalpha = p.score(alpha0,p.param['W'].W,dat,"dalpha")
        dalpha2 = 0.0*dalpha
        for i in xrange(len(alpha0)):
            alpha = alpha0.copy()
            alpha[i] = alpha[i] + h
            dalpha2[i] = (p.score(alpha,p.param['W'].W,dat,"score") - p.score(alpha0,p.param['W'].W,dat,"score"))/h
        self.assertTrue(max(abs(dalpha-dalpha2)) < self.Tol,'Derivative of score function w.r.t. alpha deviates by more than ' + str(self.Tol))


        dW = p.score(alpha0,W0,dat,"dW")
        dW2 = 0.0*dW
        
        for i in xrange(size(W0,0)):
            for j in xrange(size(W0,1)):
                W = W0.copy()
                W[i,j] = W[i,j] + h
                dW2[i,j] = (p.score(alpha0,W,dat,"score") - p.score(alpha0,W0,dat,"score"))/h
                
        self.assertTrue(max(abs(dW-dW2).flatten()) < self.Tol,'Derivative of score function w.r.t. W deviates by more than ' + str(self.Tol))
            
        

            
##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

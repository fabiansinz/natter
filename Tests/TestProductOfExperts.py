import unittest
import Distributions
import sys
from numpy.random import randn, rand
from numpy import max, abs, size, array

class TestProductOfExperts(unittest.TestCase):

    Tol = 1e-5
    
    def test_scorefct(self):
        print "Testing score function for Product of Experts ..."
        sys.stdout.flush()
        
        X = randn(5,100)
        p = Distributions.ProductOfExperts({'n':5})
        
        alpha0 = rand(p.param['N'])
    
        W0 = array(p.param['W'].W.copy())

    
        h = 1e-8
        dalpha = p.myscore(alpha0,p.param['W'].W,X,"dalpha")
        dalpha2 = 0.0*dalpha
        for i in xrange(len(alpha0)):
            alpha = alpha0.copy()
            alpha[i] = alpha[i] + h
            dalpha2[i] = (p.myscore(alpha,p.param['W'].W,X,"score") - p.myscore(alpha0,p.param['W'].W,X,"score"))/h
        self.assertTrue(max(abs(dalpha-dalpha2)) < self.Tol,'Derivative of score function w.r.t. alpha deviates by more than ' + str(self.Tol))


        dW = p.myscore(alpha0,W0,X,"dW")
        dW2 = 0.0*dW
        
        for i in xrange(size(W0,0)):
            for j in xrange(size(W0,1)):
                W = W0.copy()
                W[i,j] = W[i,j] + h
                dW2[i,j] = (p.myscore(alpha0,W,X,"score") - p.myscore(alpha0,W0,X,"score"))/h
                
        self.assertTrue(max(abs(dW-dW2).flatten()) < self.Tol,'Derivative of score function w.r.t. W deviates by more than ' + str(self.Tol))
            
        

            
##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

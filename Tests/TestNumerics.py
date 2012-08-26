import unittest
from natter.Auxiliary.Numerics import invertMonotonicIncreasingFunction, totalDerivativeOfIncGamma, inv_digamma
import mdp
from numpy.random import randn,rand
from numpy import exp, abs, amin, amax
# invertMonotonicIncreasingFunction, , 
from scipy.special import  gammainc, gamma, digamma

class TestNumerics(unittest.TestCase):
    tol = 5.0*1e-3

    def test_invertMonotonicIncreasingFunction(self):
        print "Testing invertMonotonicIncreasingFunction ..."
        a = rand()*3.0+1.
        f = lambda t: 1. / (1. + exp(-t*a))
        x = randn(10)
        y = f(x)

        xl = 0.0*x + amin(x)
        xu = 0.0*x  + amax(x)

        xstar = invertMonotonicIncreasingFunction(f,y,xl,xu)
        
        self.assertTrue(amax(abs(xstar-x)) < self.tol ,'Solution of invertMonotonicIncreasingFunction deviates more than ' + str(self.tol) + ' form the optimal value')

    def test_totalDerivativeOfIncGamma(self):
        print "Testing totalDerivativeOfIncGamma ..."
        a,b,c,d = 3.0*rand(4)+1.0
        f = lambda t: a*t + b
        df = lambda t: a + 0.0*t

        g = lambda t: c*t + d
        dg = lambda t: c + 0.0*t

        

        G = lambda x: gamma(f(x))*gammainc(f(x),g(x))
        x = rand()*2.0 + .1
        h = 1e-6
        dG = (G(x+h) - G(x))/h
        dG2 = totalDerivativeOfIncGamma(x,f,g,df,dg)
        self.assertTrue(amax(abs(dG2-dG2)) < self.tol ,'Solution of totalDerivativeOfIncGamma deviates more than ' + str(self.tol) + ' form the optimal value')
        
    def test_inv_digamma(self):
        print "Testing inv_digamma ..."
        x = rand(10)*3.0+1.0
        y = digamma(x)
        xstar = inv_digamma(y,niter=20)
        self.assertTrue(amax(abs(xstar-x)) < self.tol ,'Solution of inv_digamma deviates more than ' + str(self.tol) + ' form the optimal value')
        

##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

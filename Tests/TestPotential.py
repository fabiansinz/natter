from Auxiliary import Potential
import sys
from numpy import max, abs, log
from numpy.random import randn
from DataModule import Data
import unittest

class TestPotential(unittest.TestCase):

    potentialtypes = ['laplace','studentt']
    Tol = 1e-5
    
    
    def test_derivative(self):
        h = 1e-8
        dat = Data(randn(1,100))
        print "Testing derivatives of ..."
        for ptype in self.potentialtypes:
            print "\t" + ptype; sys.stdout.flush()
            p = Potential(ptype)
            dat2 = dat.copy()
            dat2.X = dat2.X + h

            dlogdf = p.dlogdx(dat)
            dlogdftest = (log(p(dat2)) - log(p(dat)))/h
            self.assertTrue(max(abs(dlogdftest - dlogdf)) < self.Tol, 'First derivative of potential ' + ptype + ' deviates by more than ' + str(self.Tol))

            d2logdf2 = p.d2logdx2(dat)
            d2logdf2test = (p.dlogdx(dat2) - p.dlogdx(dat))/h
            self.assertTrue(max(abs(d2logdf2test - d2logdf2)) < self.Tol, 'Second derivative of potential ' + ptype + ' deviates by more than ' + str(self.Tol))

            d3logdf3 = p.d3logdx3(dat)
            d3logdf3test = (p.d2logdx2(dat2) - p.d2logdx2(dat))/h
            self.assertTrue(max(abs(d3logdf3test - d3logdf3)) < self.Tol, 'Second derivative of potential ' + ptype + ' deviates by more than ' + str(self.Tol))
            
            
        

            
##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

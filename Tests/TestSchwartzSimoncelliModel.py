from __future__ import division
from natter import DataModule
import unittest
from natter import Distributions
from numpy import Inf, array,diag,max,abs
from numpy.random import rand
from natter.Transforms import LinearTransform
class TestSchwartzSimoncelliModel(unittest.TestCase):

    def test_derivative(self):
        W = rand(4,4)
        W = W-diag(diag(W))
        
        p = Distributions.SchwartzSimoncelliModel({'n':4,'W':LinearTransform(W),'sigma':rand()*5.0})
        print p
        dat = DataModule.DataSampler.gauss(4,10)

        dldtheta1 = p.dldtheta(dat)
        dldtheta2 = Inf*dldtheta1
        theta0 = p.primary2array()

        h = 1e-8
        Tol = 1e-6
        for k in xrange(len(theta0)):
            thetah = array(theta0)
            thetah[k] += h
            p2 = p.copy()
            p2.array2primary(thetah)
            dldtheta2[k,:] = (p2.loglik(dat)-p.loglik(dat))/h
        self.assertTrue(max(abs(dldtheta1-dldtheta2)) < Tol,'Numerical gradient deviates from returned by %d' % (Tol,)) 

    def test_restrictedDerivative(self):
        W = rand(8,8)
        W = W-diag(diag(W))
        
        p = Distributions.SchwartzSimoncelliModel(n=8,W=LinearTransform(W),sigma=rand()*5.0,restrictW=True)
        W = p['W'].W
        
        print p
        dat = DataModule.DataSampler.gauss(8,10)

        dldtheta1 = p.dldtheta(dat)
        dldtheta2 = Inf*dldtheta1
        theta0 = p.primary2array()

        h = 1e-8
        Tol = 1e-6
        for k in xrange(len(theta0)):
            thetah = array(theta0)
            thetah[k] += h
            p2 = p.copy()
            p2.array2primary(thetah)
            dldtheta2[k,:] = (p2.loglik(dat)-p.loglik(dat))/h
        self.assertTrue(max(abs(dldtheta1-dldtheta2)) < Tol,'Numerical gradient deviates from returned by %d' % (Tol,)) 
        
        

    
##################################################################################################

if __name__=="__main__":
    unittest.main()



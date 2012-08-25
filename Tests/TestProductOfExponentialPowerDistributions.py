from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from natter import Auxiliary
import sys

class TestExponentialPower(unittest.TestCase):

    TolParamP = .1
    TolParamS = .3

    def test_derivatives(self):
        print "Testing derivatives w.r.t. data ... "
        sys.stdout.flush()

        P = []
        for k in range(10):
            myp = 2.0*np.random.rand(1)[0] + .5
            mys = 3.0*np.random.rand(1)[0] + 1.0
            p = Distributions.ExponentialPower({'p':myp ,'s':mys})
            P.append(p)
            
        p = Distributions.ProductOfExponentialPowerDistributions({'P':P})

        dat = p.sample(100)
        h = 1e-7
        tol = 1e-4
        Y0 = dat.X.copy()

        df = p.dldx(dat)
        df2 = 0.0*df
        for i in xrange(dat.size(0)):
            y = Y0.copy()
            
            y[i,:] = y[i,:] + h
            df2[i,:] = (p.loglik(Data(y)) - p.loglik(dat))/h

        prot = {}
        prot['message'] = 'Difference in derivative of log-likelihood for PowerExponential greater than ' + str(tol)
        prot['max difference'] = np.max(np.abs( (df-df2).flatten()))
        prot['mean difference'] = np.mean(np.abs( (df-df2).flatten() ))

        self.assertTrue(np.max(np.abs(df-df2)) < tol,Auxiliary.prettyPrintDict(prot))


if __name__=="__main__":
    
    unittest.main()

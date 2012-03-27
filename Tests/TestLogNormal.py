from natter import Distributions
import numpy as np
import unittest
from scipy import io
import sys
from natter.DataModule import Data

class TestLogNormal(unittest.TestCase):

    Tol = 2e-2
    TolParam = 1e-1
    

    def test_estimate(self):
        print "Testing parameter estimation of LogNormal distribution ..."
        sys.stdout.flush()
        myu = 10*np.random.rand(1)[0]
        mys = 10*np.random.rand(1)[0]
        p = Distributions.LogNormal({'mu':myu ,'s':mys})
        dat = p.sample(1000000)
        p = Distributions.LogNormal()
        p.estimate(dat)
        
        self.assertFalse( np.abs(p.param['mu'] - myu) > self.TolParam,\
            'Difference in location parameter for LogNormal distribution greater than ' + str(self.TolParam))
        self.assertFalse( np.abs(p.param['s'] - mys) > self.TolParam,\
            'Difference in scale parameter for LogNormal distribution greater than ' + str(self.TolParam))

    def test_derivatives(self):
        print "Testing derivatives w.r.t. data ... "
        sys.stdout.flush()
        myu = 3.0*np.random.rand(1)[0] + 1.0
        mys = 3.0*np.random.rand(1)[0] + 1.0
        p = Distributions.LogNormal({'mu':myu ,'s':mys})
        dat = p.sample(100)
        h = 1e-7
        tol = 1e-2
        y = np.array(dat.X) + h
        df = p.dldx(dat)
        df2 = (p.loglik(Data(y)) - p.loglik(dat))/h

        self.assertFalse( np.max(np.abs(df-df2)) > tol,\
            'Difference ' + str(np.max(np.abs(df-df2)))  + ' in derivative of log-likelihood for LogNormal greater than ' + str(tol))


if __name__=="__main__":
    
    unittest.main()

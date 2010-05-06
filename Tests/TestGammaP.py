import Distribution
import Data
import numpy as np
import unittest
from Auxiliary import Errors
from scipy import io
import Auxiliary
import sys

class TestGammaP(unittest.TestCase):


    TolParam = {'p':.1, 's':.1, 'u':0.05}
    def test_estimate(self):
        print "Testing parameter estimation of GammaP distribution ..."
        sys.stdout.flush()
        myu = 10*np.random.rand(1)[0]
        mys = 10*np.random.rand(1)[0]
        myp = 1.5*np.random.rand()+.5
        p = Distribution.GammaP({'u':myu ,'s':mys,'p':myp})
        dat = p.sample(10000)
        p = Distribution.GammaP()
        p.estimate(dat)

        
        self.assertFalse(np.abs(p.param['u'] - myu) > self.TolParam,\
            'Difference in Shape parameter for Gamma distribution greater than ' + str(self.TolParam['u']))
        self.assertFalse(np.abs(p.param['s'] - mys) > self.TolParam,\
            'Difference in Scale parameter for Gamma distribution greater than ' + str(self.TolParam['s']))
        self.assertFalse(np.abs(p.param['p'] - mys) > self.TolParam,\
            'Difference in Scale parameter for Gamma distribution greater than ' + str(self.TolParam['p']))

    def test_derivatives(self):
        print "Testing derivatives w.r.t. data ... "
        sys.stdout.flush()
        myu = 10*np.random.rand(1)[0]
        mys = 10*np.random.rand(1)[0]
        myp = 1.5*np.random.rand()+.5
        p = Distribution.GammaP({'u':myu ,'s':mys,'p':myp})
        dat = p.sample(100)
        h = 1e-8
        tol = 1e-4
        y = np.array(dat.X) + h
        df = p.dldx(dat)
        df2 = (p.loglik(Data.Data(y)) - p.loglik(dat))/h
        self.assertFalse(np.max(np.abs(df-df2)) > tol,
            'Difference in derivative of log-likelihood for GammaP greater than ' + str(tol))


if __name__=="__main__":
    
    unittest.main()

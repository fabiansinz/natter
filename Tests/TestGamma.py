import Distribution
import Data
import numpy as np
import unittest
from Auxiliary import Errors
from scipy import io
import Auxiliary
import sys

class TestGamma(unittest.TestCase):

    X = Data.Data(np.array([13.335074,2.9860291,8.7888861,2.664027,5.0230222,1.9783488,1.4536823,5.0162746,8.5239465,19.658945]))
    LL = np.array([-4.0518515,-2.0986232,-2.9533653,-2.1053947,-2.2575335,-2.1744116,-2.3076855,-2.2566286,-2.8956606,-5.7716738])
    Tol = 1e-7
    TolParam = 5*1e-2
    
    def test_loglik(self):
        print "Testing log-likelihood of Gamma distribution ... "
        sys.stdout.flush()
        p = Distribution.Gamma({'u':2.0,'s':3.0})
        l = p.loglik(self.X)
        for k in range(len(self.LL)):
            self.assertFalse(np.abs(l[k] - self.LL[k]) > self.Tol,\
               'Difference in log-likelihood for Gamma greater than ' + str(self.Tol))


    def test_estimate(self):
        print "Testing parameter estimation of Gamma distribution ..."
        sys.stdout.flush()
        myu = 10*np.random.rand(1)[0]
        mys = 10*np.random.rand(1)[0]
        p = Distribution.Gamma({'u':myu ,'s':mys})
        dat = p.sample(1000000)
        p = Distribution.Gamma()
        p.estimate(dat)
        self.assertFalse(np.abs(p.param['u'] - myu) > self.TolParam,'Difference in Shape parameter for Gamma distribution greater than ' + str(self.TolParam))
        self.assertFalse(np.abs(p.param['s'] - mys) > self.TolParam,'Difference in Scale parameter for Gamma distribution greater than ' + str(self.TolParam))

    def test_derivatives(self):
        print "Testing derivatives w.r.t. data ... "
        sys.stdout.flush()
        myu = 3.0*np.random.rand(1)[0] + 1.0
        mys = 3.0*np.random.rand(1)[0] + 1.0
        p = Distribution.Gamma({'u':myu ,'s':mys})
        dat = p.sample(100)
        h = 1e-7
        tol = 1e-4
        y = np.array(dat.X) + h
        df = p.dldx(dat)
        df2 = (p.loglik(Data.Data(y)) - p.loglik(dat))/h
        self.assertFalse(np.max(np.abs(df-df2)) > tol,\
                         'Difference ' + str(np.max(np.abs(df-df2)))+ 'in derivative of log-likelihood for Gamma greater than ' + str(tol))


if __name__=="__main__":
    
    unittest.main()

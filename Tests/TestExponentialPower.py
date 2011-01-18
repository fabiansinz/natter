from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from natter.Auxiliary import Errors
from scipy import io
from natter import Auxiliary
import sys

class TestExponentialPower(unittest.TestCase):

    TolParamP = .1
    TolParamS = .3

    def test_estimate(self):
        print "Testing parameter estimation of Gamma distribution ..."
        sys.stdout.flush()
        myp = 2.0*np.random.rand(1)[0] + .5
        mys = 10.0*np.random.rand(1)[0]
        p1 = Distributions.ExponentialPower({'p':myp ,'s':mys})

        dat = p1.sample(50000)

        
        myp = 2.0*np.random.rand(1)[0] + .5
        
        mys = 10.0*np.random.rand(1)[0]
        p2 = Distributions.ExponentialPower({'p':myp ,'s':mys})
        

        p2.estimate(dat)


        prot = {}
        prot['message'] = 'Difference in parameters for Exponential Power distribution greater than threshold' 
        prot['s-threshold'] = self.TolParamS
        prot['p-threshold'] = self.TolParamP
        prot['true model'] = p1
        prot ['estimated model'] = p2
        self.assertTrue(np.abs(p2.param['p'] - p1.param['p']) < self.TolParamP or np.abs(p2.param['s'] - p1.param['s']) < self.TolParamS,\
                        Auxiliary.prettyPrintDict(prot))


    def test_derivatives(self):
        print "Testing derivatives w.r.t. data ... "
        sys.stdout.flush()
        myp = 2.0*np.random.rand(1)[0] + .5
        mys = 3.0*np.random.rand(1)[0] + 1.0
        p = Distributions.ExponentialPower({'p':myp ,'s':mys})
        dat = p.sample(100)
        h = 1e-7
        tol = 1e-4
        y = np.array(dat.X) + h
        df = p.dldx(dat)
        df2 = (p.loglik(Data(y)) - p.loglik(dat))/h


        prot = {}
        prot['message'] = 'Difference in derivative of log-likelihood for PowerExponential greater than ' + str(tol)
        prot['max difference'] = np.max(np.abs(df-df2))
        prot['mean difference'] = np.mean(np.abs(df-df2))

        self.assertTrue(np.max(np.abs(df-df2)) < tol,Auxiliary.prettyPrintDict(prot))


if __name__=="__main__":
    
    unittest.main()

import Distribution
import Data
import numpy as np
import unittest
from Auxiliary import Errors
from scipy import io
import Auxiliary
import sys



class TestLpGeneralizedNormal(unittest.TestCase):

    Tol = 1e-10
    TolParam = {'p':.1,'s':.45,'n':0}
    def test_loglik(self):
        print 'Testing log-likelihood of p-generalized normal distribution'
        sys.stdout.flush()
        for k in range(5):
            print '\t--> test case ' + str(k)
            dat = io.loadmat('/kyb/agmb/fabee/code/dev/lib/python/natter/Tests/TestPGeneralizedNormal'+ str(k) + '.mat',struct_as_record=True)
            truell = dat['ll']
            p = Distribution.LpGeneralizedNormal({'s':2*dat['s'],'p':dat['p'],'n':dat['n']})
            dat = Data.Data(dat['X'])
            ll = p.loglik(dat)
            for i in range(len(ll)):
                self.assertFalse( np.abs(ll[i]-truell[0,i]) > self.Tol,\
                    'Log-likelihood for p-generalized normal deviates from test case')

    def test_estimate(self):
        print 'Testing parameter estimation for p-generalized normal distribution'
        sys.stdout.flush()
        for k in range(5):
            print '\t--> test case ' + str(k)
            dat = io.loadmat('/kyb/agmb/fabee/code/dev/lib/python/natter/Tests/TestPGeneralizedNormal'+ str(k) + '.mat',struct_as_record=True)
            trueparam = {'s':2*dat['s'],'p':dat['p'],'n':dat['n']}
            p = Distribution.LpGeneralizedNormal({'n':dat['n']})
            dat = Data.Data(dat['X'])
            p.estimate(dat)
            for ke in trueparam.keys():
                self.assertFalse( np.abs(trueparam[ke] -  p.param[ke]) > self.TolParam[ke],\
                    'Estimated parameter ' + ke + ' deviates by more than ' + str(self.TolParam[ke]) + '!')


if __name__=="__main__":
    
    unittest.main()

from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from scipy import io
import sys
import os


class TestLpGeneralizedNormal(unittest.TestCase):

    matpath = os.path.abspath('../Tests/')

    Tol = 1e-10
    TolParam = {'p':.1,'s':.45,'n':0}
    def test_loglik(self):
        print 'Testing log-likelihood of p-generalized normal distribution'
        sys.stdout.flush()
        for k in range(5):
            print '\t--> test case ' + str(k)
            dat = io.loadmat(self.matpath + '/TestPGeneralizedNormal'+ str(k) + '.mat',struct_as_record=True)
            truell = dat['ll']
            p = Distributions.LpGeneralizedNormal({'s':2*dat['s'],'p':dat['p'],'n':dat['n']})
            dat = Data(dat['X'])
            ll = p.loglik(dat)
            for i in range(ll.shape[0]):
                self.assertFalse( np.any(np.abs(ll[i]-np.squeeze(truell[0,i])) > self.Tol),\
                    'Log-likelihood for p-generalized normal deviates from test case')

    def test_estimate(self):
        print 'Testing parameter estimation for p-generalized normal distribution'
        sys.stdout.flush()
        for k in range(5):
            print '\t--> test case ' + str(k)
            dat = io.loadmat(self.matpath + '/TestPGeneralizedNormal'+ str(k) + '.mat',struct_as_record=True)
            trueparam = {'s':2*dat['s'],'p':dat['p'],'n':dat['n']}
            p = Distributions.LpGeneralizedNormal({'n':dat['n']})
            dat = Data(dat['X'])
            p.estimate(dat)
            for ke in trueparam.keys():
                self.assertFalse( np.abs(trueparam[ke] -  p.param[ke]) > self.TolParam[ke],\
                    'Estimated parameter ' + ke + ' deviates by more than ' + str(self.TolParam[ke]) + '!')


if __name__=="__main__":
    
    unittest.main()

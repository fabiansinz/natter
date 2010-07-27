from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from scipy import io
import sys
import os

class TestLpSphericallySymmetric(unittest.TestCase):

    matpath = os.path.abspath('./Tests/')

    Tol = 1e-10
    llTol = 1e-4
    TolParam = {'p':.2,'s':.2,'u':.05}

    def test_derivatives(self):
        print "Testing derivative for p-spherically symmetric distribution with radial gamma"
        sys.stdout.flush()
        myu = 3.0*np.random.rand(1)[0]+1.0
        mys = 3.0*np.random.rand(1)[0]+1.0
        myp = 2*np.random.rand(1)[0]+.5
        n = 4
        p = Distributions.LpSphericallySymmetric({'p':myp,'n':n,'rp':Distributions.Gamma({'s':mys,'u':myu})})
        dat = p.sample(50)
        df = p.dldx(dat)
        h = 1e-8
        df2 = np.array(dat.X*np.Inf)
        for k in range(n):
            y = np.array(dat.X)
            y[k,:] += h
            df2[k,:] = (p.loglik(Data(y)) - p.loglik(dat))/h
        self.assertFalse(np.max(np.abs(df-df2).flatten()) > self.llTol,\
           'Difference ' + str(np.max(np.abs(df-df2).flatten())) + ' in derivative of log-likelihood for p-spherically symmetric greater than ' + str(self.llTol))


        print "[Ok]"

    def test_loglik(self):
        print 'Testing log-likelihood of p-spherically symmetric distribution with radial gamma'
        sys.stdout.flush()
        for k in range(5):
            print '\t--> test case ' + str(k)
            dat = io.loadmat(self.matpath + '/TestPSphericallySymmetric'+ str(k) + '.mat',struct_as_record=True)
            truell = dat['ll']
            p = Distributions.LpSphericallySymmetric({'p':dat['p'],'n':dat['n'],'rp':Distributions.Gamma({'s':dat['s'],'u':dat['u']})})
            dat = Data(dat['X'])
            ll = p.loglik(dat)
            for i in range(len(ll)):
                self.assertFalse(np.abs(ll[0,i]-truell[0,i]) > self.Tol,\
                   'Log-likelihood for p-spherically symmetric with radial gamma deviates from test case')

    def test_estimate(self):
        print 'Testing parameter estimation for p-spherically symmetric distribution with radial gamma'
        sys.stdout.flush()
        for k in range(5):
            print '\t--> test case ' + str(k)
            sys.stdout.flush()
            dat = io.loadmat(self.matpath + '/TestPSphericallySymmetric'+ str(k) + '.mat',struct_as_record=True)
            trueparam = {'s':dat['s'],'p':dat['p'],'u':dat['u']}
            p = Distributions.LpSphericallySymmetric({'n':dat['n']})
            dat = Data(dat['X'])
            p.estimate(dat,prange=(.1,4.0))

            self.assertFalse(np.abs(trueparam['p'] -  p.param['p']) > self.TolParam['p'],\
               'Estimated parameter p deviates by more than ' + str(self.TolParam['p']) + '!')
            self.assertFalse(np.abs(trueparam['u'] -  p.param['rp'].param['u']) > self.TolParam['u'],\
               'Estimated parameter u deviates by more than ' + str(self.TolParam['u']) + '!')
            self.assertFalse(np.abs(trueparam['s'] -  p.param['rp'].param['s']) > self.TolParam['s'],\
               'Estimated parameter s deviates by more than ' + str(self.TolParam['s']) + '!')



if __name__=="__main__":
    
    unittest.main()

import sys
sys.path.insert(1,'./natter')
import unittest
from natter.Distributions import Gamma, Transformed
from matplotlib.pyplot import *
from numpy.random import rand
from numpy import *
from natter.Auxiliary.Numerics import logsumexp


class TestTransformed(unittest.TestCase):


    Tol = 1e-7
    TolParam = 1e-1
    
    def setUp(self):
        self.q = Gamma(u=5.0*rand(),s=5.0*rand())
        self.p = 1.5+rand()*0.1
        def f(dat):
            dat = dat.copy()
            dat.X = dat.X**self.p
            return dat

        def finv(dat):
            dat = dat.copy()
            dat.X = dat.X**(1.0/self.p)
            return dat

        def dfinvdy(dat):
            dat = dat.copy()
            return 1.0/self.p*dat.X**( (1.0-self.p)/self.p)
        self.P = Transformed(q=self.q,f=f,finv=finv,dfinvdy=dfinvdy)
        self.TolParam = 1e-7
        
    def test_cdfppf(self):
        print "Testing consistency of cdf and ppf"
        dat = self.P.sample(100)
        err = max(abs(dat.X-self.P.ppf(self.P.cdf(dat)).X).flatten())
        self.assertTrue(err < self.TolParam,\
                         'Difference X - ppf(cdf(u)) with %.4g greater than %.4g' %(err, self.TolParam))

    def test_loglik(self):
        nsamples=100000
        q = self.q.copy()
        q['s'] = 2.0*self.q['s']
        dataImportance = q.sample(nsamples)
        # from matplotlib.pyplot import show
        # self.P.histogram(dataImportance,bins=200)
        # show()
        # raw_input()
        
        logweights = self.P.loglik(dataImportance)-q.loglik(dataImportance)
        Z = logsumexp(logweights)-log(nsamples)
        err = abs(exp(Z)-1)
        self.assertTrue(err<1e-01,'Estimated partition function deviates from 1.0 by %.4g' % (err,))
                        

        
        

if __name__=="__main__":
    
    unittest.main()

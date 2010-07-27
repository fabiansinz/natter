from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from scipy import io
from natter import Auxiliary
import sys
                               
# ##################################################################################################



##################################################################################################



class TestLpNestedSymmetric(unittest.TestCase):

    Tol = 1e-10
    llTol = 1e-4
    TolParam = {'p':.1,'s':.2,'u':.05}

    def test_derivatives(self):
        print "Testing derivative for p-nested symmetric distribution with radial gamma"
        sys.stdout.flush()
        myu = 10*np.random.rand(1)[0]
        mys = 10*np.random.rand(1)[0]
        n = 10
        L = Auxiliary.LpNestedFunction('(0,0,(1,1:4),4,(1,5:8),8:10)')
        p = Distributions.LpNestedSymmetric({'f':L,'n':n,'rp':Distributions.Gamma({'s':mys,'u':myu})})
        dat = p.sample(50)
        df = p.dldx(dat)
        h = 1e-8
        df2 = np.array(dat.X*np.Inf)
        for k in range(n):
            y = np.array(dat.X)
            y[k,:] += h
            df2[k,:] = (p.loglik(Data(y)) - p.loglik(dat))/h
        self.assertFalse(np.max(np.abs(df-df2).flatten()) > self.llTol,\
            'Difference in derivative of log-likelihood for p-nested symmetric greater than ' + str(self.llTol))

#     def test_loglik(self):
#         print "Testing log-likelihood of p-nested symmetric distribution with radial gamma",
#         print "TODO",
#         print "[Ok]"
        

    def test_estimate(self):
        print "Testing parameter estimation for p-nested symmetric distribution with radial gamma"
        sys.stdout.flush()
        L =Auxiliary.LpNestedFunction('(0,0,(1,1:3),3,(2,4:7))')
        L.p = np.random.rand(3)*1.5+.5
        
        d = Distributions.LpNestedSymmetric({'f':L,'n':L.n[()]})
        L2 =Auxiliary.LpNestedFunction('(0,0,(1,1:3),3,(2,4:7))')
        L2.p = np.random.rand(3)*1.5+.5
        L.lb = 0.0*L.p
        L.ub = 2.0*L2.p

        rd2 = Distributions.Gamma({'u':5*np.random.rand(),'s':10*np.random.rand()})
        # create Distributions object and sample
        d2 = Distributions.LpNestedSymmetric({'f':L2,'n':L2.n[()],'rp':rd2})
        print "\t ... checking greedy method"
        sys.stdout.flush()
        dat = d2.sample(50000)
        d.estimate(dat,method="greedy")
        
        self.assertFalse( np.max(np.abs(d.param['f'].p - d2.param['f'].p)) > self.TolParam['p'],\
           'Estimated parameter p deviates by more than ' + str(self.TolParam['p']) + '!')
        self.assertFalse( np.abs(d.param['rp'].param['u'] -  d2.param['rp'].param['u']) > self.TolParam['u'],\
           'Estimated parameter u deviates by more than ' + str(self.TolParam['u']) + '!')
        self.assertFalse( np.abs(d.param['rp'].param['s'] -  d2.param['rp'].param['s']) > self.TolParam['s'],\
           'Estimated parameter s deviates by more than ' + str(self.TolParam['s']) + '!')
        
        print "\t ... checking Nelder-Mead method"
        sys.stdout.flush()
        d = Distributions.LpNestedSymmetric({'f':L,'n':L.n[()]})
        d.estimate(dat,method="neldermead")


        
        self.assertFalse( np.max(np.abs(d.param['f'].p - d2.param['f'].p)) > self.TolParam['p'],\
           'Estimated parameter p deviates by more than ' + str(self.TolParam['p']) + '!')
        self.assertFalse( np.abs(d.param['rp'].param['u'] -  d2.param['rp'].param['u']) > self.TolParam['u'],\
           'Estimated parameter u deviates by more than ' + str(self.TolParam['u']) + '!')
        self.assertFalse( np.abs(d.param['rp'].param['s'] -  d2.param['rp'].param['s']) > self.TolParam['s'],\
           'Estimated parameter s deviates by more than ' + str(self.TolParam['s']) + '!')


        print "\t ... checking Gradient method"
        sys.stdout.flush()
        d = Distributions.LpNestedSymmetric({'f':L,'n':L.n[()]})
        d.estimate(dat,method="gradient")

        self.assertFalse( np.max(np.abs(d.param['f'].p - d2.param['f'].p)) > self.TolParam['p'],\
           'Estimated parameter p deviates by more than ' + str(self.TolParam['p']) + '!')
        self.assertFalse( np.abs(d.param['rp'].param['u'] -  d2.param['rp'].param['u']) > self.TolParam['u'],\
           'Estimated parameter u deviates by more than ' + str(self.TolParam['u']) + '!')
        self.assertFalse( np.abs(d.param['rp'].param['s'] -  d2.param['rp'].param['s']) > self.TolParam['s'],\
           'Estimated parameter s deviates by more than ' + str(self.TolParam['s']) + '!')

        print "[Ok]"

        
if __name__=="__main__":
    
    unittest.main()


    

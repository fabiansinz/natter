from __future__ import division
import unittest
from  natter.Distributions import ProductOfExponentialPowerDistributions, LpNestedSymmetric, LpSphericallySymmetric, LpGeneralizedNormal, Gamma
import sys
from  natter.Auxiliary import LpNestedFunction, prettyPrintDict
from natter.Transforms import NonlinearTransformFactory
import numpy as np
import sys
from numpy import linalg
from scipy import special

class TestLpNestedNonLinearICA(unittest.TestCase):

    allTol = 1e-3
    DetTol = 1e-3
    
    def test_NonlinearTransform(self):
        print "Comparing ALL before and after Nonlinear transformation ..."
        sys.stdout.flush()
        #L = LpNestedFunction('(0,0:2,(1,2:4,(2,4:6)),(2,6:8,(3,8:10)))',np.array([1.3,1.1,0.8,1.8]))
        L = LpNestedFunction()
        n = L.n[()]
        pnd = LpNestedSymmetric({'f':L,'rp':Gamma({'u':np.sqrt(n),'s':3.0*n})})

        dat = pnd.sample(50000)
        F = NonlinearTransformFactory.LpNestedNonLinearICA(pnd)

        dat2 = F*dat
        
        ica = ProductOfExponentialPowerDistributions({'n':pnd.param['f'].n[()]})
               

        ica.estimate(dat2)
        ld = F.logDetJacobian(dat)
        ld = np.mean(np.abs(ld)) / dat.size(0) / np.log(2)
        icaall = ica.all(dat2)
        pndall =  pnd.all(dat)

        
        prot = {}
        prot['message'] = 'Difference in logdet correted ALL >  ' + str(self.allTol) 
        prot["1/n/log(2) * <|det J|> "] = ld
        prot["ALL(ICA)"] = icaall
        prot["ALL(PND)"] = pndall
        prot["ALL(ICA) - 1/n/log(2) * <|det J|> - ALL(PND)"] = icaall + ld - pndall
        self.assertFalse(np.abs(icaall + ld - pndall) > self.allTol,prettyPrintDict(prot))

    def test_CheckDeterminant(self):
        print "Testing determinant ..."
        sys.stdout.flush()
        L = LpNestedFunction()
        pnd = LpNestedSymmetric({'f':L})

        dat = pnd.sample(10)
        
        F = NonlinearTransformFactory.LpNestedNonLinearICA(pnd)
        n,m = dat.size()
        h = 1e-7
        logdetJ = F.logDetJacobian(dat)
        for i in range(m):
            J = np.zeros((n,n))
            for j in range(n):
                tmp = dat[:,i]
                tmp2 = tmp.copy()
                tmp2.X[j,:] = tmp2.X[j,:] + h
                J[:,j] = ((F*tmp2).X-(F*tmp).X)[:,0]/h
            self.assertFalse( np.abs(np.log(linalg.det(J)) - logdetJ[i]) > self.DetTol,\
                              'Determinant of Jacobian deviates by more than ' + str(self.DetTol) + '!')



if __name__=="__main__":
    
    unittest.main()


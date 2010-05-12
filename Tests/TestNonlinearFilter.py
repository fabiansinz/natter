from numpy import linalg
import numpy as np
from natter import Auxiliary
import unittest
from natter.Transforms import NonlinearTransformFactory, LinearTransformFactory, LinearTransform
from numpy import linalg
from natter import Distributions
import sys

class TestNonlinearFilter(unittest.TestCase):
    
    DetTol = 1e-3

    def test_logdeterminantOfICA(self):
        print "Testing Log-Determinant of Nonlinear Lp-nested ICA ..."
        sys.stdout.flush()
        L = Auxiliary.LpNestedFunction()
        p = Distributions.LpNestedSymmetric({'f':L})
        dat = p.sample(10)
        F = NonlinearTransformFactory.LpNestedNonLinearICA(p)
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


    def test_logdeterminantInCombinationWithLinearFilters(self):
        print "Testing Log-Determinant of Nonlinear Lp-nested ICA in combination with linear filters..."
        sys.stdout.flush()
        L = Auxiliary.LpNestedFunction()
        p = Distributions.LpNestedSymmetric({'f':L})
        dat = p.sample(10)
        Flin1 = LinearTransformFactory.oRND(dat)
        Flin2 = LinearTransform(np.random.randn(dat.size(0),dat.size(0))+0.1*np.eye(dat.size(0)))
        Fnl = NonlinearTransformFactory.LpNestedNonLinearICA(p)

        Fd = {}
        Fd['NL'] = Fnl

        Fd['L1*L2'] = Flin1*Flin2
        Fd['L1*NL'] = Flin1*Fnl
        Fd['NL*L1'] = Fnl*Flin1

        Fd['Nl*L1*L2'] = Fnl*Flin1*Flin2
        Fd['Nl*(L1*L2)'] = Fnl*(Flin1*Flin2)
        Fd['(Nl*L1)*L2'] = (Fnl*Flin1)*Flin2

        Fd['L1*Nl*L2'] = Flin1*Fnl*Flin2
        Fd['L1*(Nl*L2)'] = Flin1*(Fnl*Flin2)
        Fd['(L1*Nl)*L2'] = (Flin1*Fnl)*Flin2

        Fd['L2*L1*Nl'] = Flin2*Flin1*Fnl
        Fd['L2*(L1*Nl)'] = Flin2*(Flin1*Fnl)
        Fd['(L2*L1)*Nl'] = (Flin2*Flin1)*Fnl
        
        
        
        for (tk,F) in Fd.items():
            print "\t ... testing " + tk
            sys.stdout.flush()
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
            
                self.assertFalse(np.abs(np.log(linalg.det(J)) - logdetJ[i]) > self.DetTol,\
                    'Determinant of Jacobian deviates by more than ' + str(self.DetTol) + '!')

            
##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

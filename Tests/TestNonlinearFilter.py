from numpy import linalg
import numpy as np
from natter import Auxiliary
import unittest
from natter.Transforms import NonlinearTransformFactory, LinearTransformFactory, LinearTransform
from natter import Distributions
import sys
from scipy import special

class TestNonlinearFilter(unittest.TestCase):
    
    DetTol = 1e-3

    def test_RadialFactorization(self):
        print "Testing Radial Factorization ..."
        sys.stdout.flush()
        p = np.random.rand()+1.0
        psource = Distributions.LpSphericallySymmetric({'p':p,'rp':Distributions.Gamma({'u':2.0*np.random.rand()+1.0,'s':5.0*np.random.rand()+1.0})})
        ptarget = Distributions.LpGeneralizedNormal({'p':p,'s':(special.gamma(1.0/p)/special.gamma(3.0/p))**(p/2.0)})

        F = NonlinearTransformFactory.RadialFactorization(psource)

        dat = psource.sample(10000)
        ld = F.logDetJacobian(dat)
        ld = np.mean(np.abs(ld)) / dat.size(0) / np.log(2)

        all_source = psource.all(dat)
        all_target = ptarget.all(F*dat)

        
        tol = 1e-2
        prot = {}
        prot['message'] = 'Difference in logdet correted ALL >  ' + str(tol) 
        prot["1/n/log(2) * <|det J|> "] = ld
        prot["ALL(TARGET)"] = all_target
        prot["ALL(SOURCE)"] = all_source
        prot["ALL(TARGET) + 1/n/log(2) * <|det J|> - ALL(SOURCE)"] = all_target + ld - all_source

        # -------------------------------------------------------
        # L = Auxiliary.LpNestedFunction('(0,0:2)',np.array([p]))
        # psource2 = Distributions.LpNestedSymmetric({'f':L,'n':2.0,'rp':psource.param['rp'].copy()})
        # F2 = NonlinearTransformFactory.LpNestedNonLinearICA(psource2)

        # ld2 = F2.logDetJacobian(dat)
        # ld2 = np.mean(np.abs(ld2)) / dat.size(0) / np.log(2)
        
        # all_source2 = psource2.all(dat)

        # print all_target + ld - all_source
        # print all_target + ld2 - all_source2        
        # -------------------------------------------------------

        
        self.assertTrue(np.abs(all_target + ld - all_source) < tol,Auxiliary.testProtocol(prot))

    # def test_DeterminantOfRadialFactorization(self):
    #     print "Testing Determimant of Radial Factorization ..."
    #     sys.stdout.flush()
    #     p = np.random.rand()+1.0
    #     psource = Distributions.LpSphericallySymmetric({'p':p,'rp':Distributions.Gamma({'u':2.0*np.random.rand()+1.0,'s':5.0*np.random.rand()+1.0})})

    #     # L = Auxiliary.LpNestedFunction('(0,0:2)',np.array([p]))
    #     # psource2 = Distributions.LpNestedSymmetric({'f':L,'n':2.0,'rp':psource.param['rp'].copy()})
    #     # F2 = NonlinearTransformFactory.LpNestedNonLinearICA(psource2)


    #     dat = psource.sample(100)
    #     F = NonlinearTransformFactory.RadialFactorization(psource)

    #     n,m = dat.size()
    #     h = 1e-7
    #     logdetJ = F.logDetJacobian(dat)
    #     for i in range(m):
    #         J = np.zeros((n,n))
    #         for j in range(n):
    #             tmp = dat[:,i]
    #             tmp2 = tmp.copy()
    #             tmp2.X[j,:] = tmp2.X[j,:] + h
    #             J[:,j] = ((F*tmp2).X-(F*tmp).X)[:,0]/h
    #         self.assertFalse( np.abs(np.log(linalg.det(J)) - logdetJ[i]) > self.DetTol,\
    #                           'Determinant of Jacobian deviates by more than ' + str(self.DetTol) + '!')



    # def test_logdeterminantOfICA(self):
    #     print "Testing Log-Determinant of Nonlinear Lp-nested ICA ..."
    #     sys.stdout.flush()
    #     L = Auxiliary.LpNestedFunction()
    #     p = Distributions.LpNestedSymmetric({'f':L})
    #     dat = p.sample(10)
    #     F = NonlinearTransformFactory.LpNestedNonLinearICA(p)
    #     n,m = dat.size()
    #     h = 1e-7
    #     logdetJ = F.logDetJacobian(dat)
    #     for i in range(m):
    #         J = np.zeros((n,n))
    #         for j in range(n):
    #             tmp = dat[:,i]
    #             tmp2 = tmp.copy()
    #             tmp2.X[j,:] = tmp2.X[j,:] + h
    #             J[:,j] = ((F*tmp2).X-(F*tmp).X)[:,0]/h
    #         self.assertFalse( np.abs(np.log(linalg.det(J)) - logdetJ[i]) > self.DetTol,\
    #                           'Determinant of Jacobian deviates by more than ' + str(self.DetTol) + '!')


    # def test_logdeterminantInCombinationWithLinearFilters(self):
    #     print "Testing Log-Determinant of Nonlinear Lp-nested ICA in combination with linear filters..."
    #     sys.stdout.flush()
    #     L = Auxiliary.LpNestedFunction()
    #     p = Distributions.LpNestedSymmetric({'f':L})
    #     dat = p.sample(10)
    #     Flin1 = LinearTransformFactory.oRND(dat)
    #     Flin2 = LinearTransform(np.random.randn(dat.size(0),dat.size(0))+0.1*np.eye(dat.size(0)))
    #     Fnl = NonlinearTransformFactory.LpNestedNonLinearICA(p)

    #     Fd = {}
    #     Fd['NL'] = Fnl

    #     Fd['L1*L2'] = Flin1*Flin2
    #     Fd['L1*NL'] = Flin1*Fnl
    #     Fd['NL*L1'] = Fnl*Flin1

    #     Fd['Nl*L1*L2'] = Fnl*Flin1*Flin2
    #     Fd['Nl*(L1*L2)'] = Fnl*(Flin1*Flin2)
    #     Fd['(Nl*L1)*L2'] = (Fnl*Flin1)*Flin2

    #     Fd['L1*Nl*L2'] = Flin1*Fnl*Flin2
    #     Fd['L1*(Nl*L2)'] = Flin1*(Fnl*Flin2)
    #     Fd['(L1*Nl)*L2'] = (Flin1*Fnl)*Flin2

    #     Fd['L2*L1*Nl'] = Flin2*Flin1*Fnl
    #     Fd['L2*(L1*Nl)'] = Flin2*(Flin1*Fnl)
    #     Fd['(L2*L1)*Nl'] = (Flin2*Flin1)*Fnl
        
        
        
    #     for (tk,F) in Fd.items():
    #         print "\t ... testing " + tk
    #         sys.stdout.flush()
    #         n,m = dat.size()
    #         h = 1e-7


    #         logdetJ = F.logDetJacobian(dat)
    #         for i in range(m):
    #             J = np.zeros((n,n))
    #             for j in range(n):
    #                 tmp = dat[:,i]
    #                 tmp2 = tmp.copy()
    #                 tmp2.X[j,:] = tmp2.X[j,:] + h

    #                 J[:,j] = ((F*tmp2).X-(F*tmp).X)[:,0]/h
            
    #             self.assertFalse(np.abs(np.log(linalg.det(J)) - logdetJ[i]) > self.DetTol,\
    #                 'Determinant of Jacobian deviates by more than ' + str(self.DetTol) + '!')

            


if __name__=="__main__":
    
    unittest.main()


    

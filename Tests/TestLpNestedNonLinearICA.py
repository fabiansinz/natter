import unittest
import Distributions
import sys
import Auxiliary
from Transforms import NonlinearTransformFactory
import numpy as np
import sys

class TestLpNestedNonLinearICA(unittest.TestCase):

    allTol = 1e-3
    
    def test_NonlinearTransform(self):
        print "Comparing ALL before and after Nonlinear transformation ..."
        sys.stdout.flush()
        L = Auxiliary.LpNestedFunction()
        pnd = Distributions.LpNestedSymmetric({'f':L})

        dat = pnd.sample(100000)
        
        F = NonlinearTransformFactory.LpNestedNonLinearICA(pnd)
        dat2 = F*dat
        ica = Distributions.ProductOfExponentialPowerDistributions({'n':pnd.param['f'].n[()]})
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
        prot["ALL(ICA) - 1/n/log(2) * <|det J|> - ALL(PND)"] = icaall - ld - pndall

        self.assertFalse(np.abs(icaall - ld - pndall) > self.allTol,Auxiliary.testProtocol(prot))


if __name__=="__main__":
    
    unittest.main()


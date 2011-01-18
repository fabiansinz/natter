from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from natter.Auxiliary import Errors
from scipy import io
from natter import Auxiliary
import sys

class TestMixtureOfLogNormals(unittest.TestCase):

    

    def test_derivatives(self):
        print "Testing derivatives w.r.t. data ... "
        sys.stdout.flush()
        p = Distributions.MixtureOfLogNormals({'K':5})
        dat = p.sample(100)
        h = 1e-8
        tol = 1e-4
        y = np.array(dat.X) + h
        df = p.dldx(dat)
        df2 = (p.loglik(Data(y)) - p.loglik(dat))/h

        prot = {}
        prot['message'] = 'Difference in derivative of log-likelihood for MixtureOfLogNormals greater than ' + str(tol)
        prot['max diff'] = np.max(np.abs(df-df2))
        prot['mean diff'] = np.mean(np.abs(df-df2))

        self.assertFalse(np.mean(np.abs(df-df2)) > tol, Auxiliary.prettyPrintDict(prot))

    def test_pdfloglikconsistency(self):
        print "Testing consistency of pdf and loglik  ... "
        sys.stdout.flush()

        p = Distributions.MixtureOfLogNormals({'K':5})
        dat = p.sample(100)
        
        tol = 1e-6
        ll = p.loglik(dat)
        pdf = np.log(p.pdf(dat))

        prot = {}
        prot['message'] = 'Difference in log(p(x)) and loglik(x) MixtureOfLogNormals greater than ' + str(tol)
        prot['max diff'] = np.max(np.abs(pdf-ll))
        prot['mean diff'] = np.mean(np.abs(pdf-ll))

        self.assertFalse(np.max(np.abs(ll-pdf) ) > tol,Auxiliary.prettyPrintDict(prot))


if __name__=="__main__":
    
    unittest.main()

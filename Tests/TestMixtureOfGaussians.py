from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from scipy import io
import sys

class TestMixtureOfGaussians(unittest.TestCase):

    

    def test_derivatives(self):
        print "Testing derivatives w.r.t. data ... "
        sys.stdout.flush()
        p = Distributions.MixtureOfGaussians({'K':5})
        dat = p.sample(100)
        h = 1e-7
        tol = 1e-6
        y = np.array(dat.X) + h
        df = p.dldx(dat)
        df2 = (p.loglik(Data(y)) - p.loglik(dat))/h
        self.assertFalse(np.max(np.abs(df-df2)) > tol,\
            'Difference ' +str(np.max(np.abs(df-df2))) +' in derivative of log-likelihood for MixtureOfGaussians greater than ' + str(tol))


if __name__=="__main__":
    
    unittest.main()

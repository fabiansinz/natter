from natter import Distributions
from natter.DataModule import Data
import numpy as np
import unittest
from scipy import io
import sys

class TestDirichlet(unittest.TestCase):

    X = Data(np.array([[0.1042605373,0.0443097862,0.0032503423,0.0420286884,0.1194181369,0.1848512638,0.0906818056,0.4223094329,0.4998465219,0.0078395240],[0.7299213688,0.5167476582,0.4604688785,0.4604338136,0.4221988687,0.7307655970,0.6077871086,0.1683807824,0.4403800496,0.7195288939],[0.1658180939,0.4389425556,0.5362807793,0.4975374980,0.4583829944,0.0843831391,0.3015310858,0.4093097847,0.0597734285,0.2726315821]]))
    LL = np.array([1.3689065138, 1.7564748726, 2.7472333803, 1.7160392427, 1.1921277658, 0.8798846426, 1.4837080045, -0.2267826679, -0.1300239648, 2.5635963658])
    alpha = np.array([0.6086919517, 1.9573600512, 1.3938315963])
    Tol = 1e-7
    TolParam = 5*1e-2
    
    def test_loglik(self):
        print "Testing log-likelihood of Dirichlet distribution ... "
        sys.stdout.flush()
        p = Distributions.Dirichlet({'alpha':self.alpha})
        l = p.loglik(self.X)
        for k in range(len(self.LL)):
            self.assertTrue(np.abs(l[k]- self.LL[k]) < self.Tol,'Difference in log-likelihood for Dirichlet greater than ' + str(self.Tol))


    def test_estimate(self):
        print "Testing parameter estimation of Dirichlet distribution ..."
        sys.stdout.flush()
        myalpha = 10.0*np.random.rand(10)
        p = Distributions.Dirichlet({'alpha':myalpha})
        dat = p.sample(50000)
        p = Distributions.Dirichlet({'alpha':np.random.rand(10)})
        p.estimate(dat)
        alpha = p.param['alpha']
        
        self.assertTrue(np.max(np.abs(alpha - myalpha)) < self.TolParam,'Difference in alpha parameter for Dirichlet distribution greater than ' + str(self.TolParam))




if __name__=="__main__":
    
    unittest.main()

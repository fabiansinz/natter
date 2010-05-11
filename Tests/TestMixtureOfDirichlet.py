import Distributions
import Data
import numpy as np
import unittest
from Auxiliary import Errors
from scipy import io
import Auxiliary
import sys

class TestMixtureOfDirichlet(unittest.TestCase):

    TolParam = 0.15
    


    def test_estimate(self):
        print "Testing parameter estimation of mixture of Dirichlet distributions ..."
        sys.stdout.flush()
        a1 = np.random.rand(4,2)*3.0+2.0
        pi1 = np.random.rand(2) + np.array([.1,.1])
        pi1 = pi1/np.sum(pi1)

        # change only scale
        #         a2 = np.random.rand(1)[0] * a1
        #         pi2 = pi1

        # change only mean
        #         a2 = a1 + np.random.randn(4,2)*0.5
        #         a2 = a2 * np.kron(np.ones((4,1)),np.sum(a1,0)/np.sum(a2,0))
        #         pi2 = pi1

        # change both
        a2 = a1 + np.random.rand(4,2)*.1
        pi2 = np.random.randn(2)
        pi2 = pi2/np.max(pi2)/10.0
        pi2 = pi2-np.mean(pi2)
        pi2 = pi1+pi2
        

        p = Distributions.MixtureOfDirichlet({'K':2, 'alpha':a1,'pi':pi1})
        #        p2 = Distributions.MixtureOfDirichlet({'K':2, 'alpha':a2,'pi':pi2})
        p2 = Distributions.MixtureOfDirichlet({'K':2,'n':4})


        dat = p.sample(100000)

        p2.estimate(dat,method='fixpoint',initial_guess=True)

        #         print p2
        #         p2.estimate(dat,method='bfgs',initial_guess=False)
        #         print p2
        #         raw_input()
        # p2.estimate(dat,method='kmeans')

        if np.max(np.abs(p2.param['pi'][[1,0]] - pi1.flatten())) < np.max(np.abs(p2.param['pi'] - pi1.flatten())):
            p2.param['pi'] = p2.param['pi'][[1,0]]
            p2.param['alpha'] = np.fliplr(p2.param['alpha'])
            
        
        self.assertFalse(np.max(np.abs(p2.param['alpha'].flatten() - a1.flatten())) > self.TolParam or \
               np.max(np.abs(p2.param['pi'] - pi1.flatten())) > self.TolParam,\
                         'Difference in alpha parameter for mixture of Dirichlet distributions greater than ' + str(self.TolParam))



if __name__=="__main__":
    
    unittest.main()

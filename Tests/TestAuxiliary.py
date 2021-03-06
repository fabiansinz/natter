import numpy as np
import unittest
from natter import Auxiliary
import mdp
from numpy.random import randn,rand

class TestGoldenSearch(unittest.TestCase):
    tol = 5.0*1e-2

    def test_goldeMinSearch(self):
        xl,xu = randn(2)*3.0
        h = rand()
        x0 = h*xl + (1-h)*xu
        f = lambda x: (x-x0)**2.0

        xstar = Auxiliary.Optimization.goldenMinSearch(f,xl,xu)
        xstar = .5*(xstar[0]+xstar[1])
        self.assertTrue(np.abs(xstar-x0) < self.tol ,'Solution of golden min Search deviates more than ' + str(self.tol) + ' form the optimal value')

class TestfminboundnD(unittest.TestCase):
    tol = 5.0*1e-2
    
    def test_fminboundnD(self):
        print "Testing constrained nelder-mead method ...",
        for dummy in range(10):
            n = 6
            mu = np.random.randn(n)
            x0 = np.random.randn(n)

            UB = 2.0*np.abs(mu)
            LB = -2.0*np.abs(mu)
            xt = mu.copy()
            for k in range(n):
                r = np.random.rand()
                if r < .2:
                    UB[k] = LB[k]
                    xt[k] = LB[k]
                elif r < .3:
                    UB[k] = np.Inf
                elif r < .6:
                    LB[k] = -np.Inf
                elif r < .8:
                    UB[k] = np.Inf
                    LB[k] = -np.Inf
                elif r < .9:
                    mu[k] = 2*UB[k]
                    xt[k] = UB[k]
                else:
                    mu[k] = 2*LB[k]
                    xt[k] = LB[k]

            f = lambda x: np.sum((x-mu)**2)

            xstar = Auxiliary.Optimization.fminboundnD(f,x0,LB,UB,1e-10)
            print np.max(np.abs(xstar-xt))
            self.assertTrue(np.max(np.abs(xstar-xt)) < self.tol ,'Solution of bounded Nelder-Mead deviates more than ' + str(self.tol) + ' form the optimal value')



class TestStGradient(unittest.TestCase):
    
    tolF = 1e-10

    def test_StGradient(self):
        Q0 = mdp.utils.random_rot(10)
        Q = mdp.utils.random_rot(10)
        print "Checking gradient ...",
        self.assertTrue(Auxiliary.Optimization.checkGrad(self.objective,Q,1e-4,Q0),'Mean gradient deviation is more than 1e-6')
        
        print "Checking gradient descent with Brent's method..."
        Q2 = Auxiliary.Optimization.StGradient(self.objective,Q0, {'tolT':1e-2, 'tolF':1e-8, 'maxiter':50, 'searchrange':1.0,'linesearch':'brent'},Q0)[0]
        self.assertAlmostEqual(np.sum(np.sum(np.abs(Q2-Q0))),0.0,10,'StGradient returned a result that deviated by more than ' + str(self.tolF))

        print "Checking gradient descent with golden search..."
        Q2 = Auxiliary.Optimization.StGradient(self.objective,Q0, {'tolT':1e-2, 'tolF':1e-8, 'maxiter':50, 'searchrange':1.0,'linesearch':'golden'},Q0)[0]
        self.assertAlmostEqual(np.sum(np.sum(np.abs(Q2-Q0))),0.0,10,'StGradient returned a result that deviated by more than ' + str(self.tolF))
        

    def objective(self,Q,nargout,Q0):
        sz = np.shape(Q)
        if nargout == 2:
            return (-np.sum(np.sum( (Q-Q0)**2))/float(sz[0])/float(sz[1]), -2*(Q-Q0)/float(sz[0])/float(sz[1]))
        else:
            return (-np.sum(np.sum( (Q-Q0)**2))/float(sz[0])/float(sz[1]),)



class TestPlotting(unittest.TestCase):

    def test_findshape(self):
        print "Checking findShape..."
        h, w = Auxiliary.Plotting.findShape(5*5)
        self.assertTrue(h==5 and w==5, 'findShape did not factorize 25')
        h, w = Auxiliary.Plotting.findShape(3*29)
        self.assertTrue(h==3 and w==29, 'findShape did not factorize 87')
        h, w = Auxiliary.Plotting.findShape(3*21)
        self.assertTrue(h==7 and w==9, 'findShape did not factorize 63')

        self.assertRaises( ValueError, Auxiliary.Plotting.findShape, 8951)


##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

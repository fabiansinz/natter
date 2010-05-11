import numpy as np
import unittest
from  Auxiliary import LpNestedFunction
from DataModule import Data
import sys


##################################################################################################



class TestLpNestedFunction(unittest.TestCase):

    derTol = 1e-4
    surfTol = 1e-8
    p = np.array([[0.3411156890, 1.1509880552, 0.1705872969, 1.2835449772, 2.1958492207],\
                  [2.5662500283, 2.8597702045, 1.9339300925, 2.0052164657, 1.3592980695],\
                  [1.0213538154, 1.4398143818, 2.5574954910, 1.7174933852, 2.2830570254],\
                  [1.9225209201, 0.5495975607, 2.2392212588, 0.2200682322, 2.3780081913],\
                  [2.4339028485, 2.4183806240, 2.3466016570, 2.8761309930, 1.9969234682]])
    surf = np.array([-187.1610166701,-1.5181932580, -19.0552481131, -31.7042705414, 1.1272410048])


    Tol = 10-8

    def test_functionevaluation(self):
        print "Testing function evaluation ... "
        sys.stdout.flush()
        X = np.random.randn(6,100)
        p = np.random.rand(3) + 1.0
        L = Auxiliary.LpNestedFunction('(0,0:2,(1,2:4,(2,4:6)))',p)
        dat = Data(X)
        tmp = np.sum(np.abs(X[4:6,:])**p[2],0)**(1/p[2])
        
        tmp = (tmp**p[1] + np.sum(np.abs(X[2:4,:])**p[1],0))**(1/p[1])
        tmp = (tmp**p[0] + np.sum(np.abs(X[0:2,:])**p[0],0))**(1/p[0])
        tmp2 = L.f(dat)
        self.assertFalse(np.max(np.abs(tmp-tmp2.X)) > self.Tol,\
            'Function values of Lp-nested function deviate by more that ' + str(self.Tol))
 
    def test_positiveHomogeneity(self):
        print "Testing positive homogeneity ..."
        sys.stdout.flush()
        X = np.random.randn(6,100)
        p = np.random.rand(3) + 1.0
        L = Auxiliary.LpNestedFunction('(0,0:2,(1,2:4,(2,4:6)))',p)
        a = np.random.randn()*10
        dat = Data(X)
        dat2 = Data(a*X)
        tmp = L.f(dat)
        tmp2 = L.f(dat2)
        self.assertFalse(np.max(np.abs(np.abs(a)*tmp.X-tmp2.X)) > self.Tol,\
            'Function positive homogeneity deviates by more than ' + str(self.Tol))
        

    def test_derivatives(self):
        print "Testing derivative for p-nested function ... "
        sys.stdout.flush()
        L = Auxiliary.LpNestedFunction()
        dat = Data(np.random.randn(25,100)*5.0)
        df = L.dfdx(dat)
        df2 = np.Inf*df
        h = 1e-8
        for k in range(dat.size(0)):
            Y = dat.X.copy()
            Y[k,:] += h
            dat2 = Data(Y)
            df2[k,:] = (L.f(dat2).X - L.f(dat).X)/h
        self.assertFalse(np.max(np.abs( (df2-df).flatten() )) > self.derTol,\
            'Derivatives of Lp-nested function deviate with ' + str(np.max(np.abs( (df2-df).flatten() ))) + ' by more that ' + str(self.derTol) + '!')


    def test_logSurface(self):
        print "Testing log-surface for p-nested function ..."
        sys.stdout.flush()
        L = Auxiliary.LpNestedFunction('(0, 0, (1, 1, (3, 2, 3, 4, 5), 6, 7), 8, 9, 10, 11, (2, 12, 13, 14, 15, 16, (4, 17, 18, 19, 20, 21, 22), 23), 24)')
        for k in range(len(self.surf)):
            L.p = self.p[k]
            self.assertFalse(np.abs(L.logSurface()-self.surf[k]) > self.surfTol,\
                'Log surface of Lp-nested function deviates by more than ' + str(self.surfTol) + '!')



        
if __name__=="__main__":
    
    unittest.main()


    

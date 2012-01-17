from numpy import linalg
import numpy as np
from natter.Auxiliary import Errors
import unittest
from natter.Transforms import Transform, LinearTransform, LinearTransformFactory, NonlinearTransform, NonlinearTransformFactory
from numpy import linalg, floor, array, pi, any
from natter import Distributions
from natter.DataModule import Data, DataSampler
from numpy.random import rand
import sys

from matplotlib.pyplot import show

class TestFilter(unittest.TestCase):
    
    Tol = 1e-8
    detTol = 1e-4

    def test_basics(self):
        W = LinearTransform(np.random.randn(10,10))
        print "Testing basic properties of filter:"
        print "\tMultiplication..."
        sys.stdout.flush()
        W2 = W*W

        self.assertFalse(np.max(np.abs(W2.W - W.W).flatten()) < self.Tol or np.max(np.abs(W2.W - np.dot(W.W,W.W)).flatten()) > self.Tol,'Multiplication does not work properly!')


        print "\tInversion ..."
        sys.stdout.flush()
        W2 = ~W
        self.assertFalse(np.max(np.abs(W2.W - W.W).flatten()) < self.Tol or np.max(np.abs(W2.W - linalg.inv(W.W)).flatten()) > self.Tol,'Inversion does not work properly!')


        print "\tSubsampling ..."
        sys.stdout.flush()
        W2= W[1,0:5]
        self.assertFalse(np.max(np.abs(W2.W - W.W[1,0:5]).flatten()) > self.Tol, 'Subsampling does not work properly!')
        self.assertTrue(W2.W.shape == (1,5), 'Reshaping after subsampling does not work properly!')
        

        
    def test_MaxFourier(self):
        print "Max Fourier Function "
        p = 30
        w = floor(p/2*rand(2,1))
        dat = DataSampler.gratings(p,1,w,array([2*pi*rand()]))
        F = LinearTransform(dat.X.T)
        w2 = F.getOptimalOrientationAndFrequency()
        self.assertFalse(any(w2 != w),'2D frequency of grating and max Fourier do not match!')
        

    def test_LogDetRadialTransform(self):
        print "Testing logdet of radial transformation ... "
        sys.stdout.flush()
        p = np.random.rand()*3.5+.5
        # source distribution
        psource = Distributions.LpSphericallySymmetric({'p':p})
        # target distribution
        ptarget = Distributions.LpSphericallySymmetric({'p':p,'rp':Distributions.Gamma({'u':np.random.rand()*3.0,'s':np.random.rand()*2.0})})
        # create Filter
        F = NonlinearTransformFactory.RadialTransformation(psource,ptarget)
        # sample data from source distribution
        dat = psource.sample(100)
    
        # apply filter to data
        dat2 = F*dat
        logDetJ =  F.logDetJacobian(dat)
        logDetJ2 = 0*logDetJ

        h = 1e-8

        tmp = Data(dat.X.copy())
        tmp.X[0,:] += h
        W1 = ((F*tmp).X-dat2.X)/h
        
        tmp = Data(dat.X.copy())
        tmp.X[1,:] += h
        W2 = ((F*tmp).X-dat2.X)/h
        for i in range(dat.numex()):
            
            logDetJ2[i] = np.log(np.abs(W1[0,i]*W2[1,i] - W1[1,i]*W2[0,i]))

        self.assertFalse(np.max(np.abs(logDetJ - logDetJ2)) > self.detTol,\
                         'Log determinant of radial transformation deviates by more than ' + str(self.detTol) + '!')

    def test_SubspaceEnergyWhitening(self):
        print "Testing whitening with equal subspace energy constraint..."
        n=21
        m=100000
        dat = DataSampler.gauss(n, m)
        F = LinearTransformFactory.SubspaceEnergyWhitening( dat, True )
        energy = np.diag(F.W)[1:].reshape((n-1)//2, 2)
        self.assertTrue((energy[:,0] == energy[:,1]).all(), 'energy normalization factor not identical in both subspaces')


##################################################################################################

if __name__=="__main__":
    
    unittest.main()


    

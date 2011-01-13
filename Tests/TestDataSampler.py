from __future__ import division
from natter.DataModule import DataSampler
from numpy.random import randn, rand
from numpy import pi, cos, sin, ceil, array,  any
from natter.Transforms import LinearTransform
from matplotlib.pyplot import show, plot
import unittest
from numpy.fft import fft


class TestDataSampler(unittest.TestCase):

    def test_gratings(self):
        print "Testing grating function"
        n = 100
        p = 50
        tol = 1e-4
        f = ceil(n/2*rand())
        tmp = 2*pi*rand()
        w = 3*rand()*array([cos(tmp),sin(tmp)])
        dat = DataSampler.gratings(p,n,w,array([f]))
        F = LinearTransform(randn(1,p**2))
        datout = F*dat
        z = abs(fft(datout.X))
        z[0,f] = 0
        z[0,-f] = 0
        self.assertFalse(any(z > tol),'Amplitude in temporal frequency in output other than input frequency is greater than %g' % (tol,))
       
        


if __name__=="__main__":
    
    unittest.main()

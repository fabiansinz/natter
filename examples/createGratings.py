from __future__ import division
from natter.DataModule import DataSampler
from numpy import ones, arange, pi, array, sqrt
from matplotlib.pyplot import show


if __name__=="__main__":
    w0 = 2*ones((2,))
    deltaAlpha  = arange(-pi/2,pi/2,pi/10)
    deltaOmega = array([-sqrt(2),0,sqrt(2)])
    T = 10
    phi0 = 2.0
    dat,lab = DataSampler.gratings(20,w0,2.0,deltaAlpha,deltaOmega,20)
    dat.plotPatches()
    show()

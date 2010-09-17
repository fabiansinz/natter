from __future__ import division
from natter.DataModule import DataSampler
from numpy import ones, arange, pi, array, sqrt, exp
from matplotlib.pyplot import show


if __name__=="__main__":
    w = ones((2,2))
    w[:,1] *= 2
    
    alpha  = arange(-pi/2,pi/2,pi/10)
    T = 10
    phi = 2.0
    c = exp(array([-1,0,1]))
    dat,lab = DataSampler.gratings(20,w,phi,c,alpha,10)
    print lab[alpha[0],1,c[2]]
    
    dat[:,lab[alpha[0],1,c[2]]].plotPatches()
    show()

from __future__ import division
from Gaussian import Gaussian
from numpy import hstack,resize,log,zeros

class UnnormalizedGaussian(Gaussian):
    """
    Unnormalized Gaussian distribution the normalization constant is
    set via an explicit (primary) parameter. Primary parameters are
    the parameters w.r.t which we can do inference.

    Parameters:
        n     : dimensionality
        mu    : mean
        sigma : covariance
        Z     : normalization constant

    Primary parameters:
        Z
    """

    def __init__(self,param=None):
        Gaussian.__init__(self,param)
        self.name = "Unnormalized Gaussian"
        self.param['Z'] = 10.0;
        if param.has_key('Z'):
            self.param['Z'] = param['Z']
        self.primary = ['mu','sigma','Z']

    def primary2array(self):
        arr = Gaussian.primary2array(self)
        if 'Z' in self.primary:
            arr = hstack((arr,self.param['Z']))
        return arr

    def array2primary(self,arr):
        if 'Z' in self.primary:
            Gaussian.array2primary(self,arr[:-1])
            self.param['Z'] = arr[-1]
        else:
            Gaussian.array2primary(self,arr)

    def loglik(self,data):
        lv = Gaussian.loglik(self,data)
        lv = lv - log( self.param['Z'])
        return lv
    
    def dldtheta(self,data):
        n,m = data.X.shape
        grad = Gaussian.dldtheta(self,data)
        if 'Z' in self.primary:
            if len(grad)==0:
                grad = zeros((1,m))
                grad[0,:] = -1/self.param['Z']
            else:
                grad = resize(grad,(grad.shape[0]+1,grad.shape[1]))
                grad[-1,:] = -1/self.param['Z']
        return grad

    

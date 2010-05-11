from Distributions import Gaussian
from numpy import array,hstack,squeeze

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
        self.primary = ['Z']

    def primary2array(self):
        return self.param['Z']

    def array2primary(self,arr):
        if len(arr)!=1:
            raise DimensionalityError("Normalization constant has to be scalar!")
        self.param['Z']= arr[0]
    
    def dldtheta(self,data):
        

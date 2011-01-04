from __future__ import division
from Gaussian import Gaussian
from numpy import hstack,resize,log,zeros
from copy import deepcopy

class UnnormalizedGaussian(Gaussian):
    """
    Unnormalized Gaussian distribution the normalization constant is
    set via an explicit (primary) parameter. Primary parameters are
    the parameters w.r.t which we can do inference.

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    Parameters:
        n     : dimensionality
        mu    : mean
        sigma : covariance
        Z     : normalization constant

    Primary parameters:
        Z
    """
    #@TODO: Adapt class documentation to sphinx convention.
    
    def __init__(self, *args,**kwargs):
        Gaussian.__init__(self,param)
        # parse parameters correctly
        param = None
        if len(args) > 0:
            param = args[0]
        if kwargs.has_key('param'):
            if param == None:
                param = kwargs['param']
            else:
                for k,v in kwargs['param'].items():
                    param[k] = v
        if len(kwargs)>0:
            if param == None:
                param = kwargs
            else:
                for k,v in kwargs.items():
                    if k != 'param':
                        param[k] = v
        
        # set default parameters
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

    def parameters(self,keyval=None):
        """

        Returns the parameters of the distribution as dictionary. This
        dictionary can be used to initialize a new distribution of the
        same type. If *keyval* is set, only the keys or the values of
        this dictionary can be returned (see below). The keys can be
        used to find out which parameters can be accessed via the
        __getitem__ and __setitem__ methods.

        :param keyval: Indicates whether only the keys or the values of the parameter dictionary shall be returned. If keyval=='keys', then only the keys are returned, if keyval=='values' only the values are returned.
        :type keyval: string
        :returns:  A dictionary containing the parameters of the distribution. If keyval is set, a list is returned. 
        :rtype: dict or list
           
        """
        if keyval == None:
            return deepcopy(self.param)
        elif keyval== 'keys':
            return self.param.keys()
        elif keyval == 'values':
            return self.param.value()

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

    

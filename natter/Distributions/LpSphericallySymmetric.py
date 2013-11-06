from Distribution import Distribution
from Gamma import Gamma
from natter.DataModule import Data
from numpy import log, abs, sign, squeeze, array, hstack, isscalar
from numpy.random import gamma, randn
from scipy.special import gammaln
from natter.Auxiliary.Optimization import goldenMinSearch
from copy import deepcopy


class LpSphericallySymmetric(Distribution):
    """
    Lp-spherically symmetric Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the Lp-spherically symmetric distribution
              'rp'    :  Radial distribution (default = Gamma()).
              
              'p'     :  p for the p-norm (default = 2.0)

              'n'     :  dimensionality of the data
              
    :type param: dict

    Primary parameters are ['rp','p'].
        
    """

    def __init__(self, *args,**kwargs):
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
        self.name = 'Lp-Spherically Symmetric Distribution'
        self.param = {'n':2, 'rp':Gamma(),'p':2.0}
        if param != None: 
            for k in param.keys():
                self.param[k] = param[k]
        self.param['p'] = float(self.param['p']) # make sure it is a float
        self.prange = (.1,2.0)
        self.primary = ['rp','p']

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

    def logSurfacePSphere(self):
        """
        Compute the logarithm of the surface area of the :math:`L_p`
        -norm unit sphere. This is the part of the partition function
        that is independent of x.

        :returns: The log-surface area of the :math:`L_p`-norm unit sphere.
        :rtype: float
        """
        return self.param['n']*log(2) + self.param['n']*gammaln(1/self.param['p']) \
               - gammaln(self.param['n']/self.param['p']) - (self.param['n']-1)*log(self.param['p'])

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        r = dat.norm(self.param['p'])
        return squeeze(self.param['rp'].loglik(r) \
               - self.logSurfacePSphere() - (self.param['n']-1)*log(r.X))

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the derivatives.
        :rtype:    numpy.array
        
        """        

        drdx = dat.dnormdx(self.param['p'])
        r = dat.norm(self.param['p'])
        tmp = (self.param['rp'].dldx(r) - (self.param['n']-1.0)*1.0/r.X)
        drdx *= tmp
        # for k in range(drdx.shape[0]):
        #     drdx[k,:] *= squeeze(tmp)
        return squeeze(drdx)
        
    def estimate(self,dat,prange=None):
        '''

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on
        the Distributions module <tutorial_Distributions>`).

        Fitting is carried out by performing a golden search over p
        while optimizing the radial distribution for each single p. In
        other words p* = argmax_p max_theta log(X|p,theta) where theta are
        the parameters of the radial distribution.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param prange: Range to be search in for the optimal *p*. By default prange is set to self.prange.
        :type prange:  tuple
        
        '''

        if not prange:
            prange = self.prange
        if 'p' in self.primary:
            f = lambda t: self.__pALL(t,dat)
            bestp = goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])
        if 'rp' in self.primary:
            self.param['rp'].estimate(dat.norm(self.param['p']))
        self.prange = (self.param['p']-.5,self.param['p']+.5)
            
    def __pALL(self,p,dat):
        self.param['rp'].estimate(dat.norm(p))
        self.param['p'] = p
        return self.all(dat)
        
    def sample(self,m):
        """

        Samples m samples from the current LpSphericallySymmetric distribution.

        :param m: Number of samples to draw.
        :type m: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """
        # sample from a p-generlized normal with scale 1
        z = gamma(1/self.param['p'],1.0,(self.param['n'],m))
        z = abs(z)**(1/self.param['p'])
        dat =  Data(z * sign(randn(self.param['n'],m)),'Samples from ' + self.name, \
                      ['sampled ' + str(m) + ' examples from Lp-generalized Normal'])
        # normalize the samples to get a uniform distribution.
        dat.normalize(self.param['p'])
        r = self.param['rp'].sample(m)
        dat.scale(r)
        return dat


    def primary2array(self):
        """
        Converts primary parameters into an array.

        :returns: array with primary parameters
        :rtype: numpy.ndarray
        """

        ret = array([])
        for k in self.primary:
            if k == 'p':
                ret = hstack((ret,array([self.param['p']])))
            elif k == 'rp':
                ret = hstack((ret,self.param['rp'].primary2array()))
            else:
                ret = hstack((ret,self.param[k]))
        return ret

    def array2primary(self,ar):
        """
        Converts the given array into primary parameters.

        :param ar: array containing primary parameters
        :type ar: numpy.ndarray
        :returns: The object itself.
        :rtype: natter.Distributions.Kumaraswamy

        """
        for k in self.primary:
            if k == 'p':
                self.param['p'] = ar[0]
                ar = ar[1:]
            elif k == 'rp':
                l = len(self.param['rp'].primary2array())
                self.param['rp'].array2primary(ar[:l])
                ar = ar[l:]
            else:
                if isscalar(self.param[k]):
                    self.param[k] = ar[0]
                    ar = ar[1:]
                else:
                    l = len(self.param[k])
                    self.param[k] = ar[:l]
                    ar = ar[l:]
                
                
        

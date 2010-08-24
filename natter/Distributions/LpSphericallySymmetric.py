from Distribution import Distribution
from Gamma import Gamma
from natter.DataModule import Data
from numpy import log, abs, sign
from numpy.random import gamma, randn
from scipy.special import gammaln
from natter.Auxiliary.Optimization import goldenMinSearch
from copy import deepcopy


class LpSphericallySymmetric(Distribution):
    """
    Lp-spherically symmetric Distribution

    :param param:
        dictionary which might containt parameters for the Lp-spherically symmetric distribution
              'rp'    :  Radial distribution (default = Gamma()).
              
              'p'     :  p for the p-norm (default = 2.0)
              
    :type param: dict

    Primary parameters are ['rp','p'].
        
    """

    def __init__(self,param=None):
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
               - gammaln(self.param['n']/self.param['p']) - (self.param['n']-1)*log(self.param['p']);

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        r = dat.norm(self.param['p'])
        return self.param['rp'].loglik(r) \
               - self.logSurfacePSphere() - (self.param['n']-1)*log(r.X)

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
        for k in range(len(drdx)):
            drdx[k] *= tmp
        return drdx
        
    def estimate(self,dat,prange=None):
        '''

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on
        the Distributions module <tutorial_Distributions>`).

        Fitting is carried out by alternating between a golden search
        for *p*, keeping the parameters of the radial distribution
        fixed, and optimizing the parameters of the radial
        distribution keeping the value of *p* fixed.

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
        :type name: int.
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


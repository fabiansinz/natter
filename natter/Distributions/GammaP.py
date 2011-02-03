from Gamma  import  Gamma
from natter.DataModule import Data
from numpy import log, exp, mean,squeeze
from natter import Auxiliary
from scipy.stats import gamma
from copy import deepcopy

class GammaP(Gamma):
    """
    GammaP Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the GammaP  distribution
              'p'    :    Exponent (default = 2.0)
              
              's'    :    Scale parameter (default = 1.0)

              'u'    :    Shape parameter (default = 1.0)
              
    :type param: dict

    Primary parameters are ['p','s','u'].
        
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
        self.name = 'GammaP Distribution'
        self.param = {'u':1.0, 'p':2.0, 's':1.0}
        if param!=None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['p','s','u']

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
        
    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        return squeeze(Gamma.loglik(self,dat**self.param['p']) + log(self.param['p']) + (self.param['p']-1)*log(dat.X) )

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the derivatives.
        :rtype:    numpy.array
        
        """        
        return Gamma.dldx(self,dat**self.param['p'])*self.param['p']*dat.X**(self.param['p']-1.0) \
               +(self.param['p']-1)/squeeze(dat.X)
    

    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        return exp(self.loglik(dat))
       

    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''
        return gamma.cdf(squeeze(dat.X)**self.param['p'],self.param['u'],scale=self.param['s'])

    def ppf(self,U):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        return Data(gamma.ppf(U,self.param['u'],scale=self.param['s'])**(1/self.param['p']))


    def sample(self,m):
        """

        Samples m samples from the current GammaP distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """

        dat = (Gamma.sample(self,m))**(1/self.param['p'])
        dat.setHistory([])
        return dat

    def estimate(self,dat,prange=(.1,5.0)):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).

        Estimate fits a Gamma distribution on :math:`x^p`.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param prange: Range to be search in for the optimal *p*.
        :type prange:  tuple
        
        '''

        if 'p' in self.primary:
            f = lambda t: self.__pALL(t,dat)
            bestp = Auxiliary.Optimization.goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])
        Gamma.estimate(self,dat**self.param['p'])

    def __pALL(self,p,dat):
        self.param['p'] = p
        pold = list(self.primary)
        pr= list(pold)
        if 'p' in pr:
            pr.remove('p')
        self.primary = pr
        self.estimate(dat)
        self.primary = pold
        return -mean(self.loglik(dat))





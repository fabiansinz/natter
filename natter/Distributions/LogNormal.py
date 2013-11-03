from Distribution import Distribution
from natter.DataModule import Data
from numpy import exp, log, pi, mean, std,squeeze
from numpy.random import randn
from scipy.stats import norm
from copy import deepcopy

class LogNormal(Distribution):
    """
    Log-Normal Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the Log-Normal distribution
              'mu'   :    Mean of log(x)  (default = 1.0)
              
              's'    :    Std of log(x) (default = 1.0)
              
    :type param: dict

    Primary parameters are ['mu','s'].
        
    """
    maxCount = 1000
    Tol = 10.0**-20.0
    
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
        
        self.name = 'Log-Normal Distribution'
        self.param = {'mu':0.0,'s':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['mu','s']

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
    
        
        
    def sample(self,m):
        """

        Samples m samples from the current GammaP distribution.

        :param m: Number of samples to draw.
        :type m: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """
        return Data(exp(randn(1,m)*self.param['s'] + self.param['mu']) ,str(m) + ' samples from ' + self.name)
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        return squeeze(-log(dat.X) - .5*log(pi*2.0) - log(self.param['s']) \
               - .5 / self.param['s']**2.0 * (log(dat.X) - self.param['mu'])**2)


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
        return squeeze(norm.cdf(log(dat.X),loc=self.param['mu'],scale=self.param['s']))


    def ppf(self,X):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''

        return Data(exp(norm.ppf(X,loc=self.param['mu'],scale=self.param['s'])))


    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the derivatives.
        :rtype:    numpy.array
        
        """        
        return squeeze(-1.0/dat.X  - 1.0 / self.param['s']**2.0 * (log(dat.X) - self.param['mu']) / dat.X)
        

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).

        Estimate fits the log-normal distribution by fitting a Gaussian distribution to log x.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data

        '''


        if 'mu' in self.primary:
            self.param['mu'] = mean(log(dat.X))
    

        if 's' in self.primary:
            self.param['s'] = std(log(dat.X))


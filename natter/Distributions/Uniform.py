from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros, squeeze, nan, prod
from numpy.random import rand
from natter.Auxiliary.Decorators import DataSupportChecker, ArraySupportChecker


class Uniform(Distribution):
    """
    Uniform Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.


    :param param:
        dictionary which might containt parameters for the uniform distribution
              'n'      :    dimensionality (default=1)
              'low'    :    Lower limit parameter (default = 0.0)             
              'high'   :    Upper limit parameter (default = 1.0)
              
    :type param: dict

    Primary parameters are ['low','high'].
        
    """
    maxCount = 10000
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
        self.name = 'Uniform Distribution'
        self.param = {'low':0.0,'high':1.0, 'n':1}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = []
        self.width = self.param['high'] - self.param['low']

        
    def sample(self,m):
        """

        Samples m samples from the current uniform distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return Data( rand(self.param['n'], m) * self.width + self.param['low'], \
                     str(m) + ' samples from ' + self.name)
        

    @DataSupportChecker(1,'low','high')
    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        res = log(zeros(dat.size()[1]) + 1/prod(self.width))
        return res


    @DataSupportChecker(1,'low','high')
    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        return exp(self.loglik(dat))

        
    @DataSupportChecker(1,'low','high')
    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the quantiles.
        :rtype:    numpy.array
           
        '''
        raise NotImplementedError, 'cdf not implemented in ' + self.name


    @DataSupportChecker(1,'low','high')
    def ppf(self,X):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        raise NotImplementedError, 'ppf not implemented in ' + self.name


    @DataSupportChecker(1,'low','high')
    def dldtheta(self,data):
        """
        Evaluates the gradient of the Gamma function with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        :returns:   The gradient
        :rtype:     numpy.array
        
        """

        raise NotImplementedError, 'dldtheta not implemented in ' + self.name


    @DataSupportChecker(1,'low','high')
    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array
        
        """
        raise NotImplementedError, 'dldx not implemented in ' + self.name
        
    @DataSupportChecker(1,'low','high')
    def estimate(self,dat):
        '''
        Uniform distribution has no parameter hence estimate does nothing        
        '''
        print self.name + ' has no parameter to fit.'
   
    
    def primary2array(self):
        """
        converts primary parameters into an array.
        """
        ret = zeros(len(self.primary))
        for ind,key in enumerate(self.primary):
            ret[ind]=self.param[key]
        return ret

    def array2primary(self,arr):
        """
        Converts the given array into primary parameters.
            
        """
        for ind,key in enumerate(self.primary):
            self.param[key]=arr[ind]
            
    
    
    def __setitem__(self,key,value):
        if key == 'n':
            raise NotImplementedError, 'Changing the dimensionality of ' + self.name + ' is not supported.'
        self.param[key] = value
        self.width = self.param['high'] - self.param['low']

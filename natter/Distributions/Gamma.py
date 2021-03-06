from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros, squeeze
from numpy.random import gamma
from scipy.special import gammaln, polygamma,digamma
from scipy.stats import gamma as gammastats

class Gamma(Distribution):
    """
    Gamma Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.


    :param param:
        dictionary which might containt parameters for the Gamma distribution
              'u'    :    Shape parameter, has to be >0  (default = 1.0)
              
              's'    :    Scale parameter, has to be >) (default = 1.0)
              
    :type param: dict

    Primary parameters are ['u','s'].
        
    """
    maxCount = 10000
    Tol = 10.0**-20.0

    def primaryBounds(self):
        """
        Returns bound on the primary parameters.

        :returns: bound on the primary parameters
        :rtype: list of tuples containing the specific lower and upper bound
        """

        return len(self.primary)*[(1e-6,None)]


    def __init__(self, *args,**kwargs):
        Distribution.__init__(self)
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
        self.name = 'Gamma Distribution'
        self.param = {'u':1.0,'s':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['u','s']


        
    def sample(self,m):
        """

        Samples m samples from the current Gamma distribution.

        :param m: Number of samples to draw.
        :type m: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return Data(gamma(abs(self.param['u']),abs(self.param['s']),(1,m)) \
                         ,str(m) + ' samples from ' + self.name)
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        if self.param['u'] == 1.0: # case the gamma is an exponential distribution
            return squeeze(- dat.X/abs(self.param['s'])\
                   - abs(self.param['u']) * log(abs(self.param['s'])) -  gammaln(abs(self.param['u'])) )
        else:            
            return squeeze((abs(self.param['u'])-1.0) * log(dat.X)   \
                   - dat.X/abs(self.param['s'])\
                   - abs(self.param['u']) * log(abs(self.param['s'])) -  gammaln(abs(self.param['u'])) )


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
        :returns:  A numpy array containing the quantiles.
        :rtype:    numpy.array
           
        '''
        return gammastats.cdf(squeeze(dat.X),self.param['u'],scale=self.param['s'])


    def ppf(self,X):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        return Data(gammastats.ppf(X,self.param['u'],scale=self.param['s']))


    def dldtheta(self,data):
        """
        Evaluates the gradient of the distribution with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        :returns:   The gradient
        :rtype:     numpy.array
        
        """
        m = data.size(1)
        grad = zeros((len(self.primary),m))
        for ind, pa in enumerate(self.primary):
            if pa == 'u':
                grad[ind,:] = log(data.X) - log(abs(self.param['s'])) - digamma(abs(self.param['u']))
                ind +=1
            if pa == 's':
                grad[ind,:] = data.X/abs(self.param['s']**2) - abs(self.param['u']/self.param['s'])
        return grad
     

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array
        
        """
        return squeeze((self.param['u']-1.0)/dat.X  - 1.0/self.param['s'])
        

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).

        Estimate uses the algorithm by [Minka2002]_ to fit the parameters.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''

        logmean = log(mean(dat.X))
        meanlog = mean(log(dat.X))
        u=2.0
        
        if 'u' in self.primary: # if we want to estimate u
            for k in range(self.maxCount):
                u = max(u,1e-08)
                unew= 1/u + (meanlog - logmean + log(u) - float(polygamma(0,u)))/ \
                      (u**2  * (1/u - float(polygamma(1,u))))
                unew = 1/unew
                if (unew-u)**2 < self.Tol:
                    u=unew
                    break
                u=unew
            
            self.param['u'] = unew;

        if 's' in self.primary:
            self.param['s'] = exp(logmean)/self.param['u'];
   
    
            

    

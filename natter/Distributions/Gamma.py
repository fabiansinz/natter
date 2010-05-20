from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros
from numpy.random import gamma
from scipy.special import gammaln, polygamma,digamma
from scipy.stats import gamma as gammastats


class Gamma(Distribution):
    """
    Gamma Distribution

    :param param:
        dictionary which might containt parameters for the Gamma distribution
              'u'    :    Shape parameter  (default = 1.0)
              
              's'    :    Scale parameter (default = 1.0)
              
    :type param: dict

    Primary parameters are ['u','s'].
        
    """
    maxCount = 10000
    Tol = 10.0**-20.0

    
    def __init__(self,param=None):
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
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return Data(gamma(self.param['u'],self.param['s'],(1,m)) \
                         ,str(m) + ' samples from ' + self.name)
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        return (self.param['u']-1.0) * log(dat.X)   \
               - dat.X/self.param['s']\
               - self.param['u'] * log(self.param['s']) -  gammaln(self.param['u']) 


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
        return gammastats.cdf(dat.X,self.param['u'],scale=self.param['s'])


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
        Evaluates the gradient of the Gamma function with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        
        """
        m = data.size(1)
        grad = zeros((len(self.primary),m))
        ind =0
        if 'u' in self.primary:
            grad[ind,:] = log(data.X) - log(self.param['s']) - digamma(self.param['u'])
            ind +=1
        if 's' in self.primary:
            grad[ind,:] = data.X/self.param['s']**2 - self.param['u']/self.param['s']
        return grad
     

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array
        
        """
        return (self.param['u']-1.0)/dat.X  - 1.0/self.param['s']
        

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
                unew= 1/u + (meanlog - logmean + log(u) - float(polygamma(0,u)))/ \
                      (u**2  * (1/u - float(polygamma(1,u))))
                unew = 1/unew
                if (unew-u)**2 < self.Tol:
                    u=unew
                    break
                u=unew
            
            self.param['u'] = unew;

        if 'u' in self.primary:
            self.param['s'] = exp(logmean)/self.param['u'];
   
    
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
            
    
    

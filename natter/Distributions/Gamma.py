from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean
from numpy.random import gamma
from scipy.special import gammaln, polygamma
from scipy.stats import gamma as gammastats


class Gamma(Distribution):
    """
      Gamma Distribution

      Parameters and their defaults are:
         u:    shape parameter (default = 1)
         s:    scale parameter (default = 1)
         
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
        '''

           sample(m)

           samples M examples from the gamma distribution.
           
        '''
        return Data(gamma(self.param['u'],self.param['s'],(1,m)) \
                         ,str(m) + ' samples from ' + self.name)
        

    def loglik(self,dat):
        '''

           loglik(dat)
        
           computes the loglikelihood of the data points in dat. The
           parameter dat must be a Data.Data object.
           
        '''
        return (self.param['u']-1.0) * log(dat.X)   \
               - dat.X/self.param['s']\
               - self.param['u'] * log(self.param['s']) -  gammaln(self.param['u']) 


    def pdf(self,dat):
        '''

           pdf(dat)
        
           returns the probability of the data points in dat under the
           model. The parameter dat must be a Data.Data object
           
        '''
        return exp(self.loglik(dat))
        

    def cdf(self,dat):
        '''

           cdf(dat)
        
           returns the values of the cumulative distribution function
           of the data points in dat under the model. The parameter
           dat must be a Data.Data object
           
        '''
        return gammastats.cdf(dat.X,self.param['u'],scale=self.param['s'])


    def ppf(self,X):
        '''

           ppf(X)
        
           returns the values of the inverse cumulative distribution
           function of the percentile points X under the model. The
           parameter X must be a numpy array. ppf returns a Data.Data
           object.
           
        '''
        return Data(gammastats.ppf(X,self.param['u'],scale=self.param['s']))


    def dldx(self,dat):
        """

        dldx(dat)

        returns the derivative of the log-likelihood of the gamma
        distribution w.r.t. the data in dat. The parameter dat must be
        a Data.Data object.
        
        """
        return (self.param['u']-1.0)/dat.X  - 1.0/self.param['s']
        

    def estimate(self,dat):
        '''

        estimate(dat)
        
        estimates the parameters from the data in dat (Data.Data
        object). The optional second argument specifys a list of
        parameters (list of strings) that should be estimated.
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
   
    
    

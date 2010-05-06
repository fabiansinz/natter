import Distribution
import Data
import numpy as np
import scipy as sp
from scipy import special
import mpmath
from scipy.stats import norm

class LogNormal(Distribution.Distribution):
    """
      LogNormal Distribution

      if x is log-normal distributed, then log(x) is N(mu,s)
      distributed.
      
      Parameters and their defaults are:
         mu:    location parameter (default = 0)
         s:    scale parameter (default = 1)
         
    """
    maxCount = 1000
    Tol = 10.0**-20.0
    
    def __init__(self,param=None):
        self.name = 'Log-Normal Distribution'
        self.param = {'mu':0.0,'s':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])

        
    def sample(self,m):
        '''

           sample(m)

           samples M examples from the gamma distribution.
           
        '''
        return Data.Data(np.exp(np.random.randn(1,m)*self.param['s'] + self.param['mu']) ,str(m) + ' samples from ' + self.name)
        

    def loglik(self,dat):
        '''

           loglik(dat)
        
           computes the loglikelihood of the data points in dat. The
           parameter dat must be a Data.Data object.
           
        '''
        return -np.log(dat.X) - .5*np.log(np.pi*2.0) - np.log(self.param['s']) \
               - .5 / self.param['s']**2.0 * (np.log(dat.X) - self.param['mu'])**2


    def pdf(self,dat):
        '''

           pdf(dat)
        
           returns the probability of the data points in dat under the
           model. The parameter dat must be a Data.Data object
           
        '''
        return np.exp(self.loglik(dat))
        

    def cdf(self,dat):
        '''

           cdf(dat)
        
           returns the values of the cumulative distribution function
           of the data points in dat under the model. The parameter
           dat must be a Data.Data object
           
        '''
        return norm.cdf(np.log(dat.X),loc=self.param['mu'],scale=self.param['s'])


    def ppf(self,X):
        '''

           ppf(X)
        
           returns the values of the inverse cumulative distribution
           function of the percentile points X under the model. The
           parameter X must be a numpy array. ppf returns a Data.Data
           object.
           
        '''
        return Data.Data(np.exp(norm.ppf(X,loc=self.param['mu'],scale=self.param['s'])))


    def dldx(self,dat):
        """

        dldx(dat)

        returns the derivative of the log-likelihood of the gamma
        distribution w.r.t. the data in dat. The parameter dat must be
        a Data.Data object.
        
        """
        return -1.0/dat.X  - 1.0 / self.param['s']**2.0 * (np.log(dat.X) - self.param['mu']) / dat.X
        

    def estimate(self,dat,which = None):
        '''

        estimate(dat[, which=self.param.keys()])
        
        estimates the parameters from the data in dat (Data.Data
        object). The optional second argument specifys a list of
        parameters (list of strings) that should be estimated.
        '''

        if which == None:
            which = self.param.keys()

        if which.count('mu') > 0:
            self.param['mu'] = np.mean(np.log(dat.X))
    

        if which.count('s') > 0:
            self.param['s'] = np.std(np.log(dat.X))


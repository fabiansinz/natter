import Distribution
import Data
import numpy as np
import scipy as sp
from scipy import special
import mpmath
from scipy.stats import gamma
from scipy.special import gammaln
from scipy.optimize import fminbound

class ExponentialPower(Distribution.Distribution):
    """
      Exponential Power Distribution

      Parameters and their defaults are:
         p:    skewness parameter (default = 1.0)
         s:    scale parameter (default = 1.0)
         
    """
    maxCount = 10000
    Tol = 10.0**-20.0
    
    def __init__(self,param=None):
        self.name = 'Exponential Power Distribution'
        self.param = {'p':1.0,'s':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])

        
      

    def loglik(self,dat):
        '''

           loglik(dat)
        
           computes the loglikelihood of the data points in dat. The
           parameter dat must be a Data.Data object.
           
        '''
        return np.log(self.param['p']) - np.log(2.0) - 1.0/self.param['p']*np.log(self.param['s']) \
               -gammaln(1.0/self.param['p']) - np.abs(dat.X)**self.param['p']/self.param['s']


    def dldx(self,dat):
        '''

           dldx(dat)
        
           computes the derivative of the loglikelihood w.r.t the data
           points in dat. The parameter dat must be a Data.Data
           object.
           
        '''
        return - np.sign(dat.X) * np.abs(dat.X)**(self.param['p']-1) *self.param['p'] / self.param['s']




    def sample(self,m):
        """

        sample(m)

        returns m samples from the exponential power distribution.
        
        """
        z = np.random.gamma(1.0/self.param['p'],self.param['s'],(1,m))**(1.0/self.param['p'])
        return Data.Data(np.sign(np.random.randn(1,m))*z, str(m) + ' samples from an exponential power distribution.')
        
    

    def pdf(self,dat):
        '''

           pdf(dat)
        
           returns the probability of the data points in dat under the
           model. The parameter dat must be a Data.Data object
           
        '''
        return np.exp(self.loglik(dat))
        


    def estimate(self,dat,which=None):
        if which == None:
            which = self.param.keys()

        if which.count('p') > 0:
            func = lambda t: self.__objective(t,dat,which.count('s') > 0)
            p = fminbound(func, 0.0, 100.0)[0]
            self.param['p'] = p

        if which.count('s') > 0:
            self.param['s'] = self.param['p'] * np.mean(np.abs(dat.X)**self.param['p'])
        

    def __objective(self,p,dat,est_s):
        self.param['p'] = p
        if est_s:
            self.param['s'] = p * np.mean(np.abs(dat.X)**p)
            
        return self.all(dat)

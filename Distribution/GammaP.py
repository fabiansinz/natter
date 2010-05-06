import Distribution
import Data
import numpy as np
import scipy as sp
from scipy import special
import Gamma
import Auxiliary
from scipy.stats import gamma





class GammaP(Gamma.Gamma):
    '''
      P-Gamma Distribution

      p(x) is p-gamma distributed if x^p is gamma distributed.

      Parameters and their defaults are:
         u:  shape parameter (default u=1.0)
         s:  scale parameter (default s=1.0)
         p:  exponent (default p=2)
         
    '''

    def __init__(self,param=None):
        self.name = 'P-Gamma Distribution'
        self.param = {'u':1.0, 'p':2.0, 's':1.0}
        if param!=None:
            for k in param.keys():
                self.param[k] = float(param[k])
        
    def loglik(self,dat):
        '''

           loglik(dat)
        
           computes the loglikelihood of the data points in dat. The
           parameter dat must be a Data.Data object.
           
        '''
        return Gamma.Gamma.loglik(self,dat**self.param['p']) + np.log(self.param['p']) + (self.param['p']-1)*np.log(dat.X) 

    def dldx(self,dat):
        """

        dldx(dat)

        returns the derivative of the log-likelihood of the p-gamma
        distribution w.r.t. the data in dat. The parameter dat must be
        a Data.Data object.
        
        """
        return Gamma.Gamma.dldx(self,dat**self.param['p'])*self.param['p']*dat.X**(self.param['p']-1.0) \
               +(self.param['p']-1)/dat.X
    

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
        return gamma.cdf(dat.X**self.param['p'],self.param['u'],scale=self.param['s'])

    def ppf(self,U):
        '''

           ppf(X)
        
           returns the values of the inverse cumulative distribution
           function of the percentile points X under the model. The
           parameter X must be a numpy array. ppf returns a Data.Data
           object.
           
        '''
        return Data.Data(gamma.ppf(U,self.param['u'],scale=self.param['s'])**(1/self.param['p']))


    def sample(self,m):
        '''

           sample(m)

           samples M examples from the gamma distribution.
           
        '''
        dat = (Gamma.Gamma.sample(self,m))**(1/self.param['p'])
        dat.setHistory([])
        return dat

    def estimate(self,dat,which = None,prange=(.1,5.0)):
        '''

        estimate(dat[, which=self.param.keys(),[prange=(.1,5.0)]])
        
        estimates the parameters from the data in dat (Data.Data
        object). The optional second argument specifys a list of
        parameters (list of strings) that should be estimated. prange,
        when specified, defines the search range for p.
        '''
        if which==None:
            which = self.param.keys()
        if which.count('p') > 0:
            f = lambda t: self.__pALL(t,dat)
            bestp = Auxiliary.Optimization.goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])
        Gamma.Gamma.estimate(self,dat**self.param['p'],which)

    def __pALL(self,p,dat):
        self.param['p'] = p
        self.estimate(dat,['u','s'])
        return -np.mean(self.loglik(dat))





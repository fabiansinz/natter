from __future__ import division
import Distribution
from numpy import zeros, eye, kron, dot, reshape,ones, log
from numpy.linalg import cholesky
from numpy.random import randn
from Data import Data

class Gaussian(Distribution.Distribution):
    """
      Gaussian Distribution

      Parameters and their defaults are:
         n:        dimensionality
         sigma:    covariance matrix
         mu:       mean
         
    """
    
    def __init__(self,param=None):
        self.name = 'Gaussian Distribution'
        self.param = {'n':2}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        if not param.has_key('sigma'):
            self.param['sigma'] = eye(self.param['n'])
        if not param.has_key('mu'):
            self.param['mu'] = zeros((self.param['n'],))

    
        
    def sample(self,m):
        '''
        
        sample(m)
        
        samples m examples from the gamma distribution.
    
        '''
        return Data(dot(cholesky(self.param['sigma']),randn(self.param['n'],m)) + kron(reshape(self.param['mu'],(self.param['n'],1)),ones((1,m))), \
                    str(m) + " samples from a " + str(self.param['n']) + "-dimensional Gaussian")
        

    def loglik(self,dat):
        '''
        
        loglik(dat)
        
        computes the loglikelihood of the data points in dat. The
        parameter dat must be a Data.Data object.
           
        '''

        n = self.param['n']
        C = self.param['C']
        mu = self.param['mu']
        X = dat.X
        return 

        return (self.param['u']-1.0) * np.log(dat.X)   \
               - dat.X/self.param['s']\
               - self.param['u'] * np.log(self.param['s']) -  special.gammaln(self.param['u']) 


#     def pdf(self,dat):
#         '''

#            pdf(dat)
        
#            returns the probability of the data points in dat under the
#            model. The parameter dat must be a Data.Data object
           
#         '''
#         return np.exp(self.loglik(dat))
        

#     def cdf(self,dat):
#         '''

#            cdf(dat)
        
#            returns the values of the cumulative distribution function
#            of the data points in dat under the model. The parameter
#            dat must be a Data.Data object
           
#         '''
#         return gamma.cdf(dat.X,self.param['u'],scale=self.param['s'])


#     def ppf(self,X):
#         '''

#            ppf(X)
        
#            returns the values of the inverse cumulative distribution
#            function of the percentile points X under the model. The
#            parameter X must be a numpy array. ppf returns a Data.Data
#            object.
           
#         '''
#         return Data.Data(gamma.ppf(X,self.param['u'],scale=self.param['s']))


#     def dldx(self,dat):
#         """

#         dldx(dat)

#         returns the derivative of the log-likelihood of the gamma
#         distribution w.r.t. the data in dat. The parameter dat must be
#         a Data.Data object.
        
#         """
#         return (self.param['u']-1.0)/dat.X  - 1.0/self.param['s']
        

#     def estimate(self,dat,which = None):
#         '''

#         estimate(dat[, which=self.param.keys()])
        
#         estimates the parameters from the data in dat (Data.Data
#         object). The optional second argument specifys a list of
#         parameters (list of strings) that should be estimated.
#         '''

#         if which == None:
#             which = self.param.keys()

#         logmean = np.log(np.mean(dat.X))
#         meanlog = np.mean(np.log(dat.X))
#         u=2.0

#         if ( which.count('u') > 0): # if we want to estimate u
#             for k in range(self.maxCount):
#                 unew= 1/u + (meanlog - logmean + np.log(u) - float(special.polygamma(0,u)))/ \
#                       (u**2  * (1/u - float(special.polygamma(1,u))))
#                 unew = 1/unew
#                 if (unew-u)**2 < self.Tol:
#                     u=unew
#                     break
#                 u=unew
            
#             self.param['u'] = unew;

#         if which.count('s') > 0:
#             self.param['s'] = np.exp(logmean)/self.param['u'];
   
    
    

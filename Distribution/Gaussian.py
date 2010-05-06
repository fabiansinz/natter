from __future__ import division
import Distribution
from numpy import zeros, eye, kron, dot, reshape,ones, log,pi, sum, diag, exp, where, triu, hstack, squeeze, array
from numpy.linalg import cholesky, inv
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

        self.primary = ['mu','sigma']
        self.I =  where(triu(ones((self.param['n'],self.param['n'])))>0)
        
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
        m = dat.size(1)
        C = self.param['sigma']
        mu = self.param['mu']
        X = dat.X - kron(reshape(mu,(n,1)),ones((1,m)))
        return -n/2*log(2*pi) -sum(log(diag(cholesky(C)))) - .5* sum(X*dot(inv(C),X),0)

    def pdf(self,dat):
        '''
        
        pdf(dat)
        
        returns the probability of the data points in dat under the
        model. The parameter dat must be a Data.Data object
        
        '''
        return exp(self.loglik(dat))
    

#     def tri2flat(self, A = None):
#         if A == None:
#             A = self.C.copy()
#         return np.squeeze(A[self.I])
        
#     def flat2tri(self,a,A=None):
#         if A == None:
#             A = self.C.copy()
       
#         A[self.I] = a
#         A = np.triu(A) + np.triu(A,1).transpose()
#         return A
        


    def primary2array(self):
        ret = array([])
        if self.primary.count('mu') > 0:
            ret = hstack((ret,self.param['mu'].copy()))
        if self.primary.count('sigma') > 0:
            ret = hstack((ret,squeeze(self.param['sigma'][self.I])))
        return ret


    def array2primary(self,pr):
        n = self.param['n']
        if self.primary.count('mu') > 0:
            self.param['mu'] = pr[:n]
            pr = pr[n:]
        if self.primary.count('sigma') > 0:
            C = zeros((n,n))
            C[self.I] = pr
            C = triu(C) + np.triu(C,1).transpose()
            self.param['sigma'] = C



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
   
    
    

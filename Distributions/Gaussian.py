from __future__ import division
from Distributions import Distribution
from numpy import zeros, eye, kron, dot, reshape,ones, log,pi, sum, diag, exp, where, triu, hstack, squeeze, array,vstack,outer
from numpy.linalg import cholesky, inv
from numpy.random import randn
import Data 
from Auxiliary.Errors import AbstractError

class Gaussian(Distribution):
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
        self.I =  where(tril(ones((self.param['n'],self.param['n'])))>0)
        self.cholP = cholesky(inv(self.param['sigma']))
        
    def sample(self,m):
        '''
        
        sample(m)
        
        samples m examples from the gamma distribution.
    
        '''
        return Data.Data(dot(cholesky(self.param['sigma']),randn(self.param['n'],m)) + kron(reshape(self.param['mu'],(self.param['n'],1)),ones((1,m))), \
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
    
    def primary2array(self):
        ret = array([])
        if 'mu' in self.primary:
            ret = hstack((ret,self.param['mu'].copy()))
        if 'sigma' in self.primary:
            ret = hstack((ret,squeeze(self.param['sigma'][self.I])))
        return ret


    def array2primary(self,pr):
        n = self.param['n']
        if 'mu' in self.primary:
            self.param['mu'] = pr[:n]
            pr = pr[n:]
        if 'sigma' in self.primary:
            C = zeros((n,n))
            C[self.I] = pr
            C = triu(C) + triu(C,1).transpose()
            self.param['sigma'] = C


    def dldtheta(self,dat):
        ret = array([])
        n,m = dat.size()
        if 'mu' in self.primary:
            ret = dot(inv(self.param['sigma']),dat.X - reshape(self.param['mu'],(n,1)))

        if 'sigma' in self.primary:
            CI = inv(self.param['sigma'])
            retC = zeros((len(self.I[0]),m));
            for i,x in enumerate(dat.X.T):
                X = x - self.param['mu']
                X = outer(X,X)
                C = -CI + dot(dot(CI,X),CI)
                C = C - .5 * diag(diag(C))
                retC[:,i] = C[self.I] 
            ret = vstack((ret,retC))
        return ret

   
    
    
from __future__ import division
from Distribution import Distribution
from numpy import zeros, eye, kron, dot, reshape,ones, log,pi, sum, diag, exp, where, triu, tril, hstack, squeeze, array,vstack,outer
from numpy.linalg import cholesky, inv, solve
from numpy.random import randn
from natter.DataModule import Data 
from natter.Auxiliary import debug


class Gaussian(Distribution):
    """
    Gaussian Distribution

    Base class for the Gaussian distribution.  
    
    :arguments:
        param : dictionary which might containt parameters for the Gaussian
              'n'    :    dimensionality (default=2)
              'sigma':    covariance matrix (default = eye(dimensionality))
              'mu'   :    mean  (default = zeros(dimensionality))

    *Example*

        >>> normal = Gaussian( {'n':5} )
        >>> normal.sample(10)
        ------------------------------
        Data object: 10 samples from a 5-dimensional Gaussian
	10  Examples
	5  Dimensions
        ------------------------------
        
    """
    
    def __init__(self,param=None):
        self.name = 'Gaussian Distribution'
        self.param = {'n':2}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        if not self.param.has_key('sigma'):
            self.param['sigma'] = eye(self.param['n'])
        if not self.param.has_key('mu'):
            self.param['mu'] = zeros((self.param['n'],))
        self.primary = ['mu','sigma']
        self.I =  where(tril(ones((self.param['n'],self.param['n'])))>0)
        # internally, we represent the covariance in terms of the cholesky factor of the precision matrix
        self.cholP = cholesky(inv(self.param['sigma'])) 
        
    def sample(self,m):
        '''
        
        sample(m)
        
        samples m examples from the Gaussian distribution.
    
        '''
        return Data(dot(cholesky(self.param['sigma']),randn(self.param['n'],m)) + kron(reshape(self.param['mu'],(self.param['n'],1)),ones((1,m))), \
                    str(m) + " samples from a " + str(self.param['n']) + "-dimensional Gaussian")
        

    def loglik(self,dat):
        '''
        
        loglik(dat)
        
        computes the loglikelihood of the data points in dat. The
        parameter dat must be a natter.DataModule.Data object.

        :return:
            array of log-likelihood values for each data-point in dat
           
        '''

        n = self.param['n']
        m = dat.size(1)
        C = self.param['sigma']
        mu = self.param['mu']
        X = dat.X - kron(reshape(mu,(n,1)),ones((1,m)))
        X = dot(self.cholP,X)
        X = diag(dot(X.T,X))
        return -n/2*log(2*pi) +sum(log(diag(self.cholP))) - .5*X

    def pdf(self,dat):
        '''
        
        pdf(dat)
        
        returns the probability of the data points in dat under the
        model. The parameter dat must be a Data.Data object
        
        '''
        return exp(self.loglik(dat))
    
    def primary2array(self):
        """
        :return:
        array containing primary parameters. 
        """
        ret = array([])
        if 'mu' in self.primary:
            ret = hstack((ret,self.param['mu'].copy()))
        if 'sigma' in self.primary:
            ret = hstack((ret,squeeze(self.cholP[self.I])))
        return ret


    def array2primary(self,arr):
        n = self.param['n']
        if 'mu' in self.primary:
            self.param['mu'] = arr[:n]
            arr = arr[n:]
        if 'sigma' in self.primary:
            self.cholP[self.I] = arr
            self.param['sigma'] = solve(self.cholP,solve(self.cholP.T,eye(n)))


    def dldtheta(self,dat):
        ret = array([])
        n,m = dat.size()
        if 'mu' in self.primary:
            ret = solve(self.cholP,solve(self.cholP.T, dat.X -reshape(self.param['mu'],(n,1))))
        if 'sigma' in self.primary:
            v = diag(1/diag(self.cholP))[self.I]
            retC = zeros((len(self.I[0]),m));
            for i,x in enumerate(dat.X.T):
                X = x - self.param['mu']
                X = outer(X,X)
                retC[:,i] = -dot(self.cholP.T,X)[self.I]   + v
            if len(ret)==0:
                ret = retC
            else:
                ret = vstack((ret,retC))
        return ret

   
    def estimate(self,dat):
        if 'sigma' in self.primary:
            self.param['sigma'] = dat.cov()
        if 'mu' in self.primary:
            self.param['mu'] = dat.mean()
            
        
    

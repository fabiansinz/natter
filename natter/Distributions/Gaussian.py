from __future__ import division
from Distribution import Distribution
from numpy import zeros, eye, kron, dot, reshape,ones, log,pi, sum, diag,  where,  tril, hstack, squeeze, array,vstack,outer
from numpy.linalg import cholesky, inv, solve
from numpy.random import randn
from natter.DataModule import Data 
#from natter.Auxiliary import debug
from scipy import optimize

class Gaussian(Distribution):
    """
    Gaussian Distribution

    Base class for the Gaussian distribution.  
    
    :param param:
        dictionary which might containt parameters for the Gaussian
              'n'    :    dimensionality (default=2)
              
              'sigma':    covariance matrix (default = eye(dimensionality))
              
              'mu'   :    mean  (default = zeros(dimensionality))

    :type param: dict

    Primary parameters are ['mu','sigma'].
        
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
        """

        Samples m samples from the current Gaussian distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return Data(dot(cholesky(self.param['sigma']),randn(self.param['n'],m)) + kron(reshape(self.param['mu'],(self.param['n'],1)),ones((1,m))), \
                    str(m) + " samples from a " + str(self.param['n']) + "-dimensional Gaussian")
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''

        n = self.param['n']
        m = dat.size(1)
        C = self.param['sigma']
        mu = self.param['mu']
        X = dat.X - kron(reshape(mu,(n,1)),ones((1,m)))
        Y = sum(dot(self.cholP.T,X)**2,axis=0)
        return -n/2*log(2*pi) +sum(log(diag(abs(self.cholP)))) - .5*Y

    
    def primary2array(self):
        """
        :return: array containing primary parameters. If 'Sigma' is in
        the primary parameters, then the cholesky factor of the
        precision matrix is filled in the array.
        """
        ret = array([])
        if 'mu' in self.primary:
            ret = hstack((ret,self.param['mu']))
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
            self.cholP[where(eye(n)>0)] = map(lambda x: max(x,1e-08),diag(self.cholP))
            self.param['sigma'] = solve(self.cholP.T,solve(self.cholP,eye(n)))


    def dldtheta(self,dat):
        """
        Calculates the gradient with respect to the primary
        parameters. Note: if 'Sigma' is in the primary parameters the
        gradient is calculated with respect to the cholesky factor of
        the inverse covariance matrix.
        
        """
        ret = array([])
        n,m = dat.size()
        if 'mu' in self.primary:
            ret = solve(self.cholP,solve(self.cholP.T, dat.X -reshape(self.param['mu'],(n,1))))
        if 'sigma' in self.primary:
            v = diag(1.0/diag(self.cholP))[self.I]
            retC = zeros((len(self.I[0]),m));
            for i,x in enumerate(dat.X.T):
                X = x - self.param['mu']
                X = outer(X,X)
                retC[:,i] = -dot(self.cholP.T,X).T[self.I]   + v
            if len(ret)==0:
                ret = retC
            else:
                ret = vstack((ret,retC))
        return ret
        

    def estimate(self,dat,method=None):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).


        :param dat: Data points on which the Gaussian distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''

        if method==None:
            method = "analytic"
        if method=="analytic":
            if 'sigma' in self.primary:
                self.param['sigma'] = dat.cov()
                self.cholP = cholesky(inv(self.param['sigma'])) 
            if 'mu' in self.primary:
                self.param['mu'] = dat.mean()
        else:
            def f(arr):
                self.array2primary(arr)
                return -sum(self.loglik(dat))
            def df(arr):
                self.array2primary(arr)
                return -sum(self.dldtheta(dat),axis=1)
            arr0 = self.primary2array()
            arropt = optimize.fmin_bfgs(f,arr0,df)
                
            
        
    

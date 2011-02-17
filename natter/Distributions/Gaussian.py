from __future__ import division
from Distribution import Distribution
from numpy import zeros, eye, kron, dot, reshape,ones, log,pi, sum, diag,  where,  tril, hstack, squeeze, array,vstack,outer,sqrt
from numpy.linalg import cholesky, inv, solve
from numpy.random import randn
from natter.DataModule import Data 
#from natter.Auxiliary import debug
from scipy import optimize
from copy import deepcopy
from scipy.linalg import solve_triangular
from scipy.stats import norm 
from natter.Auxiliary.Errors import DimensionalityError

class Gaussian(Distribution):
    """
    Gaussian Distribution

    Base class for the Gaussian distribution.  

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    
    :param param:
        dictionary which might containt parameters for the Gaussian
              'n'    :    dimensionality (default=2)
              
              'sigma':    covariance matrix (default = eye(dimensionality))
              
              'mu'   :    mean  (default = zeros(dimensionality))

    :type param: dict

    Primary parameters are ['mu','sigma'].
        
    """

    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = None
        if len(args) > 0:
            param = args[0]
        if kwargs.has_key('param'):
            if param == None:
                param = kwargs['param']
            else:
                for k,v in kwargs['param'].items():
                    param[k] = v
        if len(kwargs)>0:
            if param == None:
                param = kwargs
            else:
                for k,v in kwargs.items():
                    if k != 'param':
                        param[k] = v
        
        # set default parameters
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
        # internally, we represent the covariance in terms of the
        # cholesky factor of the precision matrix
        self.cholP = cholesky(inv(self.param['sigma'])) 

    def parameters(self,keyval=None):
        """

        Returns the parameters of the distribution as dictionary. This
        dictionary can be used to initialize a new distribution of the
        same type. If *keyval* is set, only the keys or the values of
        this dictionary can be returned (see below). The keys can be
        used to find out which parameters can be accessed via the
        __getitem__ and __setitem__ methods.

        :param keyval: Indicates whether only the keys or the values of the parameter dictionary shall be returned. If keyval=='keys', then only the keys are returned, if keyval=='values' only the values are returned.
        :type keyval: string
        :returns:  A dictionary containing the parameters of the distribution. If keyval is set, a list is returned. 
        :rtype: dict or list
           
        """
        if keyval == None:
            return deepcopy(self.param)
        elif keyval== 'keys':
            return self.param.keys()
        elif keyval == 'values':
            return self.param.value()
        
    def sample(self,m):
        """

        Samples m samples from the current Gaussian distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return Data(solve(self.cholP,randn(self.param['n'],m)) + kron(reshape(self.param['mu'],(self.param['n'],1)),ones((1,m))), \
                    str(m) + " samples from a " + str(self.param['n']) + "-dimensional Gaussian")

        # return Data(dot(cholesky(self.param['sigma']),randn(self.param['n'],m)) + kron(reshape(self.param['mu'],(self.param['n'],1)),ones((1,m))), \
        #             str(m) + " samples from a " + str(self.param['n']) + "-dimensional Gaussian")
        

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
        mu = self.param['mu']
        X = dat.X - kron(reshape(mu,(n,1)),ones((1,m)))
        Y = sum(dot(self.cholP.T,X)**2,axis=0)
        return -n/2*log(2*pi) +sum(log(diag(abs(self.cholP)))) - .5*Y

    
    def primary2array(self):
        """
        :return: array containing primary parameters. If 'sigma' is in
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
            #self.cholP[where(eye(n)>0)] = map(lambda x: max(x,1e-08),diag(self.cholP))
            self.param['sigma'] = solve(self.cholP.T,solve(self.cholP,eye(n)))

    def __setitem__(self,key,value):
        Distribution.__setitem__(self,key,value)
        if key == 'mu':
            self.param['mu'] = value
        if key == 'sigma':
            self.param['sigma'] = value
            self.cholP = solve_triangular(cholesky(self.param['sigma']).T,eye(self['n'])).T
        if key == 'CholP':
            self.cholP = value
            self.param['sigma'] = solve(self.cholP.T,solve(self.cholP,eye(self.param['n'])))
    def dldtheta(self,dat):
        """
        Calculates the gradient with respect to the primary
        parameters. Note: if 'sigma' is in the primary parameters the
        gradient is calculated with respect to the cholesky factor of
        the inverse covariance matrix.
        
        """
        ret = array([])
        n,m = dat.size()
        if 'mu' in self.primary:
            ret = dot(self.cholP,dot(self.cholP.T, dat.X -reshape(self.param['mu'].copy(),(n,1))))
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
            optimize.fmin_bfgs(f,arr0,df)
                
            
        
    
    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        Works only for n=1.

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the quantiles.
        :rtype:    numpy.array
           
        '''
        if self.param['n'] == 1:
            return squeeze(norm.cdf(dat.X,loc=self.param['mu'],scale=sqrt(self.param['sigma'])))
        else:
            raise DimensionalityError('Gaussian cdf only works for n=1!')


    def ppf(self,u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        if self.param['n'] == 1:
            return Data(norm.ppf(u,loc=self.param['mu'],scale=sqrt(self.param['sigma'])),"Percentiles from a Gaussian distribution")
        else:
            raise DimensionalityError('Gaussian ppf only works for n=1!')

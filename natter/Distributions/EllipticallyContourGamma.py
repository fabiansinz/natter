from __future__ import division
from Gamma import Gamma
from natter.DataModule import Data
from CompleteLinearModel import CompleteLinearModel
from natter.Transforms import LinearTransform
from scipy.special import gammaln
from numpy import log,sum,array,dot,kron,ones,vstack,hstack,pi,sqrt,where,tril,diag,zeros, eye, outer
from numpy.random import randn
from mdp.utils import random_rot
from copy import deepcopy
from scipy.optimize import fmin_bfgs




class EllipticallyContourGamma(CompleteLinearModel):
    """
    Ellipticallly contoured distribution with a gamma radial distribution.
    It is a special case of the complete linear model.

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    
    :math: `p(x) \sim \Gamma(||Wx||_2|\alpha,\beta) det(W) `
    
    """

    def __init__(self,  *args,**kwargs):
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
        if param is None:
            self.param = {}
        else:
            self.param =param
        if not  self.param.has_key('q'):
            self.param['q']=Gamma()
        if not self.param.has_key('n'):
            self.param['n']=2
        if not self.param.has_key('W'):
            self.param['W']=LinearTransform(random_rot(self.param['n']),\
                                                      'Random rotation matrix',['sampled from Haar distribution'])
        self.name = 'Elliptically contour Gamma distribution'
        self.Wind = where(tril(ones(self.param['W'].W.shape))>0)
        self.param['W'].W = tril(self.param['W'].W)
        if not self.param.has_key('primary'):
            self.primary=['q','W']
        else:
            self.primary=param['primary']
        
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

        
    def primary2array(self):
        """
        Converts primary parameters to an array.

        :returns: array with primary parameters
        :rtype: numpy.ndarray
        """
        ret = array([])
        if 'q' in self.primary:
            ret = self.param['q'].primary2array()
        if 'W' in self.primary:
            if len(ret)==0:
                ret = self.param['W'].W[self.Wind].flatten()
            else:
                ret = hstack((ret,self.param['W'].W[self.Wind].flatten()))
        return ret

    def sample(self,m):
        """
        Sample from an elliptically contoured Gamma distribution.
        
        To do so, we first sample from a Gaussian distribution with
        the appropiate Covariance matrix, normalize and then draw a
        radius from the radial-Gamma distribution.

        :param m: number of samples
        :type m: int
        :returns: samples from the distribution
        :rtype: natter.DataModule.Data
        """

        WX = dot(self.param['W'].W,randn(self.param['W'].W.shape[0],nsamples))
        r  = self.param['q'].sample(nsamples)
        return Data(WX/sqrt(sum(WX**2,axis=0))*r.X)
    
    def array2primary(self,arr):
        """
        Converts array to primary parameters.

        :param arr: array with primary parameters
        :type arr: numpy.ndarray
        """
        if 'q' in self.primary:
            self.param['q'].array2primary(arr[0:2])
            arr = arr[2:]
        if 'W' in self.primary:
            self.param['W'].W[self.Wind] = arr#.reshape(self.param['W'].W.shape)
    
        
    def loglik(self,data):
        """
        Computes the log likelihood of the given data and returns it
        for each data point individually.

        :param data: data
        :type data: natter.DataModule.Data

        :returns: array of log-likelihood values for each data point in data.
        :rtype: numpy.array
        """
        n,m = data.size()
        squareData = self.param['W']*data
        squareData.X = sqrt(sum(squareData.X**2,axis=0))
        y = self.param['q'].loglik(squareData) + \
            array(data.size(1)*[sum(log(abs(diag(self.param['W'].W))))])\
            -(n/2)*log(pi) +gammaln(n/2) -log(2) + (1-n)*log(squareData.X)
        return y

        
        
        
    def dldtheta(self,data):
        """
        Calculates the gradient with respect to the primary parameters on the data set given.

        :param data: data points at which the derivative is computed
        :type data: natter.DataModule.Data
        :returns: array with derivatives
        :rtype: numpy.ndarray
        """
        ret = array([])
        n,m = data.size()
        squareData = self.param['W']*data
        squareData.X = sqrt(sum(squareData.X**2,axis=0))
        for pa in self.primary:
            if pa == 'q':
                gradG = self.param['q'].dldtheta(squareData)
                ret0 = gradG
            if pa== 'W':
                u = self.param['q'].param['u']
                s = self.param['q'].param['s']
                W = self.param['W'].W
                wx2 = squareData.X
                v = diag(1.0/diag(W))[self.Wind] # d(log(det))/dW
                WXXT    = zeros((n,n,m))
                code = """
                for (int i=0;i<n;i++){
                   for (int j=0;j<n;j++){
                      for (int l=0;l<m;l++){
                         for (int u=0;u<n;u++){
                            WXXT(i,j,l) += W(i,u)*X(u,l)*X(j,l);
                        }
                      }
                   }
                }
                """
                X = data.X
                try:
                    from scipy import weave
                    from scipy.weave import converters
                    weave.inline(code,
                                 ['W', 'X', 'WXXT', 'n','m'],
                                 type_converters=converters.blitz,
                                 compiler = 'gcc')
                    WXXT = WXXT[self.Wind[0],self.Wind[1],:]
                except Exception, e:
                    print "Failed to compile inline code.\nTraceback: %s\n\nFalling back to slow mode!"%(e)
                WXXT    = zeros((len(v),m))
                for k in xrange(m):
                    WXXT[:,k]= dot(W,outer(data.X[:,k],data.X[:,k]))[self.Wind]
                        
              
                
                gradW = ((u-n)/wx2  -1/s)*WXXT*(1/wx2)  +kron(ones((m,1)),v).T# kron(ones((m,1)),inv(W.T).flatten()).T
                ret0 = gradW
            if len(ret)==0:
                ret = ret0
            else:
                ret = vstack((ret,ret0))
        return ret
        
                
    def estimate(self,data):
        """
        Estimate the primaray parameters of the distribution based on  gradient descent.

        :param data: data points from which the primary parameters are estimated
        :type data: natter.DataModule.Data
        """

        x0 = self.primary2array()
        def f(x):
            old = self.primary2array()
            self.array2primary(x)
            L = -sum(self.loglik(data))
            self.array2primary(old)
            return L
        def df(x):
            old = self.primary2array()
            self.array2primary(x)
            g = -self.dldtheta(data)
            self.array2primary(old)
            return g
        xopt = fmin_bfgs(f,x0,fprime=df,disp=0)
        self.array2primary(xopt)
        
        

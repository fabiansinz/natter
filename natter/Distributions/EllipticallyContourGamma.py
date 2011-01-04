from __future__ import division
from Gamma import Gamma
from natter.DataModule import Data
from CompleteLinearModel import CompleteLinearModel
from scipy.special import gammaln
from numpy import log,sum,array,dot,kron,ones,vstack,hstack,pi,sqrt,where,tril,diag,zeros
from numpy.random import randn

from scipy import weave
from scipy.weave import converters
from copy import deepcopy





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
        """
        
        """
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

        gamma   = Gamma()
        param['q'] = gamma
        CompleteLinearModel.__init__(self,param)
        if 'q' in self.primary:
            self.param['q'].primary = ['u','s']
        self.name = 'Elliptically contour Gamma distribution'
        self.Wind = where(tril(ones(self.param['W'].W.shape))>0)
        self.param['W'].W = tril(self.param['W'].W)

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
        ret = array([])
        if 'q' in self.primary:
            ret = self.param['q'].primary2array()
        if 'W' in self.primary:
            if len(ret)==0:
                ret = self.param['W'].W[self.Wind].flatten()
            else:
                ret = hstack((ret,self.param['W'].W[self.Wind].flatten()))
        return ret

    def sample(self,nsamples=1):
        """
        Sample from an elliptically contoured Gamma distribution.
        
        To do so, we first sample from a Gaussian distribution with
        the appropiate Covariance matrix, normalize and then draw a
        radius from the radial-Gamma distribution.
        
        """

        WX = dot(self.param['W'].W,randn(self.param['W'].W.shape[0],nsamples))
        r  = self.param['q'].sample(nsamples)
        return Data(WX/sqrt(sum(WX**2,axis=0))*r.X)
    
    def array2primary(self,arr):
        """
        Converts array to primary parameters
        """
        if 'q' in self.primary:
            self.param['q'].array2primary(arr[0:2])
            arr = arr[2:]
        if 'W' in self.primary:
            self.param['W'].W[self.Wind] = arr#.reshape(self.param['W'].W.shape)
    
        
    def loglik(self,data):
        """
        
        Arguments:
        - `self`:
        - `data`:
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

        
        """
        ret = array([])
        n,m = data.size()
        squareData = self.param['W']*data
        squareData.X = sqrt(sum(squareData.X**2,axis=0))
        if 'q' in self.primary:
            gradG = self.param['q'].dldtheta(squareData)
            ret = gradG
        if 'W' in self.primary:
            u = self.param['q'].param['u']
            s = self.param['q'].param['s']
            W = self.param['W'].W
            wx2 = squareData.X
            v = diag(1.0/diag(W))[self.Wind] # d(log(det))/dW
            # WXXT    = zeros((len(v),m))
            # for k in xrange(m):
            #     WXXT[:,k]= dot(W,outer(data.X[:,k],data.X[:,k]))[self.Wind]
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
            weave.inline(code,
                         ['W', 'X', 'WXXT', 'n','m'],
                         type_converters=converters.blitz,
                         compiler = 'gcc')
            WXXT = WXXT[self.Wind[0],self.Wind[1],:]
            #print "difference norm: ",norm(WXXT-WXXT2)

            
            
            gradW = ((u-n)/wx2  -1/s)*WXXT*(1/wx2)  +kron(ones((m,1)),v).T# kron(ones((m,1)),inv(W.T).flatten()).T
            if len(ret)==0:
                ret = gradW
            else:
                ret = vstack((ret,gradW))
        return ret
        
                
    def estimate(self,data):
        """
        Estimate the primaray parameters of the distribution based on  gradient descent.
        """
        

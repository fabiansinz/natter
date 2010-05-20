from __future__ import division
from Distribution import Distribution
from Gamma import Gamma
from natter.DataModule import Data
from CompleteLinearModel import CompleteLinearModel
from scipy.special import gammaln,digamma
from numpy import log,exp,sum,array,dot,outer,kron,ones,vstack,hstack,pi,sqrt
from numpy.linalg import inv







class EllipticallyContourGamma(CompleteLinearModel):
    """
    Ellipticallly contoured distribution with a gamma radial distribution.
    It is a special case of the complete linear model.
    
    :math: `p(x) \sim \Gamma(||Wx||_2|\alpha,\beta) det(W) `
    
    """

    def __init__(self, param=None):
        """
        
        """
        gamma   = Gamma()
        param['q'] = gamma
        CompleteLinearModel.__init__(self,param)
        if 'q' in self.primary:
            self.param['q'].primary = ['u','s']
        self.name = 'Elliptically contour Gamma distribution'

    def primary2array(self):
        ret = array([])
        if 'q' in self.primary:
            ret = self.param['q'].primary2array()
        if 'W' in self.primary:
            if len(ret)==0:
                ret = self.param['W'].W.flatten()
            else:
                ret = hstack((ret,self.param['W'].W.flatten()))
        return ret


    def array2primary(self,arr):
        """
        Converts array to primary parameters
        """
        if 'q' in self.primary:
            self.param['q'].array2primary(arr[0:2])
            arr = arr[2:]
        if 'W' in self.primary:
            self.param['W'].W = arr.reshape(self.param['W'].W.shape)
    
        
    def loglik(self,data):
        """
        
        Arguments:
        - `self`:
        - `data`:
        """
        n,m = data.size()
        squareData = self.param['W']*data
        squareData.X = sqrt(sum(squareData.X**2,axis=0))
        y = self.param['q'].loglik(squareData) + self.param['W'].logDetJacobian()\
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
            WXXT = array( map(lambda x: dot(W,outer(x,x)).flatten(),data.X.T )).T
            gradW = ((u-n)/wx2  -1/s)*WXXT*(1/wx2)  + kron(ones((m,1)),inv(W.T).flatten()).T
            if len(ret)==0:
                ret = gradW
            else:
                ret = vstack((ret,gradW))
        return ret
        
                
    def estimate(self,data):
        """
        Estimate the primaray parameters of the distribution based on  gradient descent.
        """
        

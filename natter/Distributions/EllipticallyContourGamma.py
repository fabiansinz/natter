from __future__ import division
from Distribution import Distribution
from Gamma import Gamma
from natter.DataModule import Data
from CompleteLinearModel import CompleteLinearModel
from numpy.special import gammaln,digamma
from numpy import log,exp,sum,array,dot,outer,kron,ones,vstack,hstack
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


    def primary2array(self):
        ret = array([])
        if 'q' in self.primary:
            ret = self.param['q'].primary2array()
        if 'w' in self.primary:
            if len(ret)==0:
                ret = self.param['w'].W.flatten()
            else:
                ret = hstack((ret,self.param['W']))
        return ret
                        
        
    def loglik(self,data):
        """
        
        Arguments:
        - `self`:
        - `data`:
        """

        squareData = self.param['W']*data
        squareData.X = sum(squareData.X**2,axis=0)
        y = self.param['q'].loglik(squareData) + self.param['W'].logDetJacobian
        return y


    def dldtheta(self,data):
        """
        Calculates the gradient with respect to the primary parameters on the data set given.

        
        """
        ret = array([])
        n,m = data.size()
        squareData = Data()
        squareData.X = sum((self.param['W']*data.X).X**2,axis=0)
        if 'q' in self.primary:
            gradG = self.param['q'].dldtheta(squareData)
            ret = gradG
        if 'W' in self.primary:
            u = self.param['q'].param['u']
            s = self.param['q'].param['s']
            W = self.param['W'].W
            wx2 = squareData.X
            gradW = ((u-1)/wx2  -1/s)*array( map(lambda x: dot(W,outer(x,x)).flatten(),data.X.T ))  + kron(ones((m,1)),inv(W).flatten()).T
            if len(ret)==0:
                ret = gradW
            else:
                ret = vstack((ret,gradW))
        return ret
        
                

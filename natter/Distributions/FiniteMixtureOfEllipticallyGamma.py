from __future__ import division
from FiniteMixtureDistribution import FiniteMixtureDistribution
from EllipticallyContourGamma import EllipticallyContourGamma
from natter.DataModule import Data
from numpy.linalg import cholesky,inv,solve
from numpy import zeros,ones,dot,eye,log,mean,exp,sqrt,cov,sum,var,isnan
from mdp.utils import random_rot,symrand
from natter.Transforms import LinearTransform
from natter.Auxiliary.Numerics import logsumexp
import sys

class FiniteMixtureOfEllipticallyGamma(FiniteMixtureDistribution):
    """
    Class for representing a finite mixture of elliptically Gamma-distributions.

    That is, the radial component is Gamma-distributed.
    
    """

    def __init__(self, param=None):
        """
        """

        if param==None:
            param = {'NC':2}
        if 'n' in param.keys():
            self.param['n']= param['n']
        else:
            self.param['n']=2
            
        if 'NC' in param.keys():
            self.numberOfMixtureComponents= param['NC']
            self.ps = zeros(self.numberOfMixtureComponents,dtype=object)
        if 'q' in param.keys():
            for p in self.ps:
                p = param['q'].copy()
        else:
            for k in xrange(self.numberOfMixtureComponents):
                W =symrand(self.param['n'])
                W = dot(W,W.T)
                self.ps[k] = EllipticallyContourGamma({'n': self.param['n'],
                                                       'W': LinearTransform(cholesky(W))})
        self.alphas = ones(self.numberOfMixtureComponents)/self.numberOfMixtureComponents
        self.primary = ['alpha','theta']
        
        
    def estimate(self,data,method=None):
        """
        Estimate the parameters of each ECG distribution using the EM
        algorithm by default.

        :Arguments:
           :param data: Data object to fit the mixture on.
           :type data: DataModule.Data

           :param method: Optional parameter to choose the method to
                          use for fitting
           :type method: String

        :return: No return values, the optimal parameters are already
                 set.
        
        """
        if method==None or method=="EM":
            n,m = data.size()
            K   = self.numberOfMixtureComponents
            T = zeros((K,m))
            LP = zeros((K,m))
            done = False
            diff = 100
            oldLP = 10000
            while not done:
                for k in xrange(K):
                    LP[k,:] = self.ps[k].loglik(data)  + log(self.alphas[k])
                for k in xrange(K):
                    T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))
                self.alphas = mean(T,axis=1) # mstep
                for k in xrange(K):
                    TS = sum(T[k,:])
                    X = data.X
                    X = X*exp(0.5*(log(T[k,:]) -log(TS) + log(m)))
                    C = cov(X) + eye(n)*1e-05
                    if isnan(C).any():
                        C = eye(n)
                    # C = cov(X)*(m-1) + eye(n)*1e-05 # add a ridge
                    Y = Data(sqrt(sum(dot(self.ps[k].param['W'].W,X)**2,axis=0)))
                    if 'q' in self.ps[k].primary:
                        self.ps[k].param['q'].estimate(Y)

                    if 'W' in self.ps[k].primary:
                        self.ps[k].param['W'].W =  solve(cholesky(C),eye(n))

                cALL=sum(-(T*LP).flatten())/(n*m)/log(2)
                diff = abs(oldLP-cALL)/abs(oldLP) # relative difference...
                print "\rrelative difference: " ,diff , "  current ALL: " , cALL ," ",
                sys.stdout.flush()
                oldLP = cALL
                if diff<1e-08:
                    done=True
                



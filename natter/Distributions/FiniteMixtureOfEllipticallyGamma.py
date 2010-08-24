from __future__ import division
from FiniteMixtureDistribution import FiniteMixtureDistribution
from EllipticallyContourGamma import EllipticallyContourGamma
from natter.DataModule import Data
from numpy.linalg import cholesky,inv,solve
from numpy import zeros,ones,dot,eye,log,mean,exp,sqrt,cov,sum,var,isnan,outer
from mdp.utils import random_rot,symrand
from natter.Transforms import LinearTransform
from natter.Auxiliary.Numerics import logsumexp
import sys
from scipy import optimize
from natter.Auxiliary import profileFunction
from copy import deepcopy
from scipy import weave
from scipy.weave import converters


class FiniteMixtureOfEllipticallyGamma(FiniteMixtureDistribution):
    """
    Class for representing a finite mixture of elliptically Gamma-distributions.

    That is, the radial component is Gamma-distributed.
    
    """

    def __init__(self, param=None):
        """
        """
        FiniteMixtureDistribution.__init__(self)
        if param==None:
            param = {'NC':2}
        if 'n' in param.keys():
            self.param['n']= param['n']
        else:
            self.param['n']=2
            
        if 'NC' in param.keys():
            self.param['numberOfMixtureComponents']= param['NC']
            self.param['ps'] = zeros(self.param['numberOfMixtureComponents'],dtype=object)
        if 'q' in param.keys():
            for p in self.param['ps']:
                p = param['q'].copy()
        else:
            for k in xrange(self.param['numberOfMixtureComponents']):
                W =symrand(self.param['n'])
                W = dot(W,W.T)
                self.param['ps'][k] = EllipticallyContourGamma({'n': self.param['n'],
                                                       'W': LinearTransform(cholesky(W))})
        self.param['alphas'] = ones(self.param['numberOfMixtureComponents'])/self.param['numberOfMixtureComponents']
        self.primary = ['alpha','theta']
        
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

        
    def estimate(self,data,method=None,tol=1e-08):
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
        n,m = data.size()
        K   = self.param['numberOfMixtureComponents']
        T = zeros((K,m))
        LP = zeros((K,m))
        done = False
        diff = 100
        oldLP = 10000
        warnflag=0
        while not done:
            for k in xrange(K):
                LP[k,:] = self.param['ps'][k].loglik(data)  + log(self.param['alphas'][k])
            for k in xrange(K):
                T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))
            self.param['alphas'] = mean(T,axis=1) # mstep
            if method=="EM" or method==None:
                for k in xrange(K):
                    TS = sum(T[k,:])
                    X = data.X
                    #X = X*exp(0.5*(log(T[k,:]) -log(TS) + log(m)))
                    C = zeros((n,n))
                    code = """
                           for (int g=0;g<m;g++){
                                for (int i=0;i<n;i++){
                                    for (int j=i;j<n;j++){
                                       C(i,j)=C(j,i)+=X(i,g)*X(j,g)*U(k,g);
                                    }
                                }
                            }
                    """
                    U=T/TS;
                    weave.inline(code,
                                 ['C', 'X', 'U', 'k', 'n','m'],
                                 type_converters=converters.blitz,
                                 compiler = 'gcc')

                    # for l in xrange(m):
                    #     C = C + outer(X[:,l],X[:,l])*T[k,l]/TS
                    C = C + eye(n)*1e-02
                    if isnan(C).any():
                        print "Uiuiui"
                        C = eye(n)
                    # C = cov(X)*(m-1) + eye(n)*1e-05 # add a ridge
                    Y = Data(sqrt(sum(dot(self.param['ps'][k].param['W'].W,X)**2,axis=0)))
                    if 'q' in self.param['ps'][k].primary:
                        self.param['ps'][k].param['q'].estimate(Y)

                    if 'W' in self.param['ps'][k].primary:
                        self.param['ps'][k].param['W'].W =  solve(cholesky(C),eye(n))
            else:
                self.primary=['theta']
                def f(arr):
                    self.array2primary(arr)
                    return -sum(self.loglik(data))
                def df(arr):
                    self.array2primary(arr)
                    return -sum(self.dldtheta(data),axis=1)

                arr0=self.primary2array()
                xopt, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = optimize.fmin_bfgs(f,arr0,df,maxiter=2,disp=0,full_output=1)
                
            cALL=self.all(data)
            diff = abs(oldLP-cALL)/abs(oldLP) # relative difference...
            print "\rrelative difference: " ,diff , "  current ALL: " , cALL ," ",
            sys.stdout.flush()
            oldLP = cALL
            if diff<tol or warnflag==2:
                done=True

                        



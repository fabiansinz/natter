from __future__ import division
from Distribution import Distribution
from Gaussian import Gaussian
from numpy import zeros,ones,log,exp,array,vstack,hstack,sum,mean,kron,isnan,dot,unique, reshape, where,cumsum, squeeze, Inf, max, abs
from numpy.random.mtrand import multinomial
from natter.DataModule import Data
from numpy.random import shuffle
import sys
from scipy import optimize
#from functions import minimize_carl
from copy import deepcopy
from natter.Auxiliary.Utils import parseParameters
from natter.Auxiliary.Errors import ValueError
from natter.Auxiliary.Numerics import logsumexp
from warnings import warn
from natter.Auxiliary.Decorators import Squeezer

def logistic(eta):
    return 1/(1+exp(-eta))

class FiniteMixtureDistribution(Distribution):
    """
    Base class for a finite mixture of base distributions.

    :math:`p(x|\\theta) = \sum_{k=1}^{K} \\alpha_k p(x|\\theta_k)`


    The constructor is either called with a dictionary, holding the
    parameters (see below) or directly with the parameter assignments
    (e.g. myDistribution(n=2,b=5)). Mixed versions are also possible.


    :param param:
        dictionary which might containt parameters for the Gamma distribution
              'P'    :    List of compatible Distribution objects determining the single mixture components.

              'alpha'   :    Numpy array containing the mixture proportions. It must sum to one. 
              
    :type param: dict

    Primary parameters are ['P','alpha'].  

    """

    def __init__(self,*args, **kwargs):
        # set initial parameters
        param = parseParameters(args,kwargs)
        self.param = {}
        for k,v in param.items():
            self[k] = v
        
        if not self.param.has_key('P'):
            raise ValueError('You must specify the parameter P!')

        
        K = len(self.param['P'])
        if not self.param.has_key('alpha'):
            self['alpha'] = ones(K)/K
        self.name = "Finite mixture of %s distributions" % (", ".join(unique(array([p.name for p in self['P']],dtype=object))),)
        self.primary = ['alpha','P']

        
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
            return self.param.values()   


    def __setitem__(self,k,v):
        self.param[k] = deepcopy(v)

    def sample(self,m):
        """

        Samples m samples from the current finite mixture distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        dim = self['P'][0].sample(1).dim()
        nc = multinomial(m,self.param['alpha'])
        mrange = range(m)
        shuffle(mrange)
        X = zeros((dim,m))
        ind = 0
        K = len(self['P'])
        for k in xrange(K):
            dat = self.param['P'][k].sample(nc[k])
            X[:,mrange[ind:ind + nc[k]]] = dat.X
            ind += nc[k]
        return Data(X,"%i samples from a %i-dimensional finite mixture distribution" % (m,dim))

    def checkAlpha(self):
        s=0.0
        for alpha in self.param['alpha']:
            ab = alpha
            alpha = max(min(alpha,1.0-s),0.0)
            s +=alpha
            diff =abs(ab-alpha)
            if diff > 1e-03:
                print "Warning : diff in alphas: ", diff



    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        self.checkAlpha()
        n,m = dat.size()
        X = zeros((m,len(self.param['P'])))
        for k,p in enumerate(self.param['P']):
            X[:,k] = p.loglik(dat) + log(self.param['alpha'][k])
        return logsumexp(X,axis=1)

    def pdf(self,dat):
        '''
        
        Evaluates the probability density function on the data points in dat. 
        
        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
        
        '''
        return exp(self.loglik(dat))

    @Squeezer(1)
    def ppf(self,u,bounds=None,maxiter=1000):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given
        array of quantiles. The single mixture components must
        implement ppf and pdf.

        NOTE: ppf works only for one dimensional mixture distributions.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :param bounds: a tuple of two array of the same size of u that specifies the initial upper and lower boundaries for the bisection method.
        :type bounds: tuple of two numpy.array 
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''

        ret = Data(u,'Percentiles from ' + self.name)
        # use bisection method on to invert
        #v = squeeze(log(u/(1-u)))
        if bounds is not None:
            lb = Data(bounds[0])
            ub = Data(bounds[1])
        elif self.param['P'][0].param.has_key('a') and self.param['P'][0].param.has_key('b'):
            warn("\tAssuming that the keys a and b in %s refer to boundaries. Using those..." % (self.param['P'][0].name,))
            lb = Data(0*u+self.param['P'][0]['a'])
            ub = Data(0*u+self.param['P'][0]['b'])
        else:
            lb = Data(u*0-1e6)
            ub = Data(u*0+1e6)
        def f(dat):
            # c = self.cdf(dat)
            # return v - log(c/(1-c))
            return u-self.cdf(dat)

        iterC = 0
        while max(ub.X-lb.X) > 5*1e-10 and iterC < maxiter:
            ret.X = (ub.X+lb.X)/2
            mf = f(ret)
            lf = f(lb)
            uf = f(ub)
            if any(lf*uf>0):
                warn("ppf lost the root! resetting boundaries")
                ind0 = where(lf*uf > 0)
                ub.X[0,ind0[0]] = 4*abs(ub.X[0,ind0[0]]+1)
                lb.X[0,ind0[0]] = -4*abs(lb.X[0,ind0[0]]+1)
            ind0 = where(mf*lf < 0)
            ind1 = where(mf*uf < 0)
            ub.X[0,ind0[0]] = ret.X[0,ind0[0]]
            lb.X[0,ind1[0]] = ret.X[0,ind1[0]]
            iterC +=1
            # sys.stdout.write(80*" " + "\r\tFiniteMixtureDistribution.ppf maxdiff: %.4g, meandiff: %.4g" % (max(ub.X-lb.X),mean(ub.X-lb.X)))
            # sys.stdout.flush()
        if iterC == maxiter:
            warn("FiniteMixtureDistribution.ppf: Maxiter reached! Exiting. Bisection method might not have been converged. Maxdiff is %.10g. Mean diff is %.4g" % ( max(ub.X-lb.X),mean(ub.X-lb.X)))
        #sys.stdout.write("\n")
        return ret


    def primary2array(self):
        ret = array([])
        if 'alpha' in self.primary:
            ret = hstack((ret,log(self.param['alpha'][:-1]/(1.0- self.param['alpha'][:-1]))))
        if 'P' in self.primary:
            for k in xrange(len(self.param['P'])):
                ret = hstack((ret,self.param['P'][k].primary2array()))
        return ret

    def array2primary(self,arr):
        K = len(self.param['P'])
        if 'alpha' in self.primary:
            if any(arr[:K] > 40) or any(arr[:K] < -40):
                warn("FiniteMixtureDistribution.array2primary: values of alpha to extreme. fixing that")
                arr[:K][where(arr[:K] > 30)] = 30
                arr[:K][where(arr[:K] < -30)] = -30
                self.param['alpha'][0:K-1]= logistic(arr[0:K-1])
                self.param['alpha'][-1]=1-sum(self.param['alpha'][0:K-1])
                self.param['alpha'] = self.param['alpha']/sum(self.param['alpha'])
            else:
                self.param['alpha'][0:K-1]= logistic(arr[0:K-1])
                self.param['alpha'][-1]=1-sum(self.param['alpha'][0:K-1])
                
            arr = arr[K-1::]
        if 'P' in self.primary:
            for k in xrange(K):
                lp = len(self.param['P'][k].primary2array())
                self.param['P'][k].array2primary(arr[0:lp])
                arr = arr[lp::]

    def dldtheta(self,dat):
        K = len(self.param['P'])
        self.checkAlpha()
        lp = self.loglik(dat)
        n,m = dat.size()
        ret = array([])
        etas = log(self.param['alpha'][:-1]/(1.0- self.param['alpha'][:-1]))
        if 'alpha' in self.primary:
            ret=zeros((K-1,m))
            dldp = zeros((K,m))
            dpdeta = zeros((K-1,K))
            for k in xrange(K-1):
                dldp[k,:] = exp(self.param['P'][k].loglik(dat)-lp)
                dpdeta[k,k] = logistic(etas[k])*logistic(-etas[k])
            dldp[-1,:] = exp(self.param['P'][-1].loglik(dat)-lp)
            dpdeta[:,-1]= -logistic(etas)*logistic(-etas)
            ret = dot(dpdeta,dldp)
        if 'P' in self.primary:
            ret0 = self.param['P'][0].dldtheta(dat)
            lp0  = self.param['P'][0].loglik(dat)
            ret0 = ret0*exp(lp0-lp + log(self.param['alpha'][0]))
            for k in xrange(1,K):
                ret1 = self.param['P'][k].dldtheta(dat)
                lp1  = self.param['P'][k].loglik(dat)
                ret1 = ret1*exp(lp1-lp + log(self.param['alpha'][k]))
                ret0 = vstack((ret0,ret1))
            if len(ret)==0:
                ret = ret0
            else:
                ret = vstack((ret,ret0))
        return ret

    def cdf(self,dat):
        """

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the quantiles.
        :rtype:    numpy.array
           
        """
        u = self.param['P'][0].cdf(dat)*self.param['alpha'][0]
        
        for k in xrange(1,len(self.param['P'])):
            u += self.param['P'][k].cdf(dat)*self.param['alpha'][k]
        return u
            

    def estimate(self,dat,method=None,maxiter=100):
        """


        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on  the Distributions module <tutorial_Distributions>`).

        Additionally a method for fitting the distribution can be
        specified. 'EM' performs standard EM algorithm, the
        E-step is analytical, the M-step is done via gradient descent.
        'gradient' tries to directly perform a gradient ascent method
        on the (joint) likelihood. 'hybrid' first performs EM until
        initial convergence and switches to gradient afterwards.


        :param dat: Data points on which the distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param method: method to fit the distribution. The choice is between 'EMGradient', 'gradient' or 'hybrid'
        :type method: string

           
        """
        K = len(self.param['P'])
        if method==None:
            method = 'EM'
            print "Using: %s-method " % (method,)

        if method=="EM" or method == 'hybrid':
            n,m = dat.size()
            T = zeros((K,m)) # alpha(i)*p_i(x|theta)/(sum_j alpha(j) p_j(x|theta))
            LP = zeros((K,m)) # log likelihoods of the single mixture components

            def estep():
                for k in xrange(K):
                    LP[k,:] = self.param['P'][k].loglik(dat)  + log(self.param['alpha'][k])
                for k in xrange(K):
                    T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))
                return sum((T*LP).flatten())

            def mstep():
                n,m = dat.size()
                def f(ar):
                    par = ar.copy()
                    L = T.copy()
                    for k in xrange(K):
                        mg = len(self.param['P'][k].primary2array())
                        self.param['P'][k].array2primary(par[:mg])
                        par = par[mg:]
                        X = self.param['P'][k].loglik(dat)
                        L[k,:] = T[k,:]*X
                    sys.stdout.write(80*" " + "\r" + 5*"\t" + "current ALL : %.10g"% (-mean(L.flatten()),))
                    sys.stdout.flush()
                    return -sum(L.flatten())/K/m
                def df(ar):
                    par = ar.copy()
                    grad = zeros(len(ar))
                    ind =0
                    for k in xrange(K):
                        mg = len(self.param['P'][k].primary2array())
                        self.param['P'][k].array2primary(par[0:mg])
                        par = par[mg:]
                        dX = self.param['P'][k].dldtheta(dat)
                        grad[ind:ind+mg] = sum(T[k,:]*dX,axis=1)
                        ind += mg
                    return -grad/K/m
                def check(arr):
                    err = optimize.check_grad(f,df,arr)
                    print "Error in gradient: ",err
                    sys.stdout.flush()
                
                arr = self.primary2array()
                if 'alpha' in self.primary:
                    arr = arr[K-1:] # because alpha are reparametrized and only the first K-1 are returned
                #check(arr)
                optimize.fmin_bfgs(f,arr,df,gtol=1e-3,disp=0,full_output=0)
                    
                if 'alpha' in self.primary:
                    self.param['alpha'] = mean(T,axis=1)


            
            diff = 10000000
            oldS = estep()
            tol = 1e-7
            if method == 'hybrid': tol = tol**.25

            iterC= 0
            while abs(diff)>tol and iterC < maxiter:
                mstep()
                fv= estep()
                diff = oldS-fv
                oldS=fv
                sys.stdout.write(80*" " +  "\r\tDiff: %.4g\t" % (diff,))
                sys.stdout.flush()
                iterC += 1
            if iterC == maxiter:
                sys.stdout.write("\n\tEM maxiter reached! EM might not have converged!")
            sys.stdout.write("\n")
            sys.stdout.flush()
                
        if method=='gradient' or method=='hybrid':
            n,m = dat.size()
            
            def f(arr):
                self.array2primary(arr)
                LL=-sum(self.loglik(dat))
                sys.stdout.write(80*" "+ "\r\tnLL: %.10g \t\t\tcurrent ALL: %.10g" % (LL/(m*n),self.all(dat)))
                sys.stdout.flush()
                return LL/(m*n)
            
            def df(arr):
                self.array2primary(arr)
                grad = sum(self.dldtheta(dat),axis=1)
                return -grad/(m*n)

            arr0 = self.primary2array()
            diff = 100000;
            arr0 = self.primary2array()

            optimize.fmin_bfgs(f,arr0,fprime=df,maxiter=maxiter,gtol=1e-10)
            


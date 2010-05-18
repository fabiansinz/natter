from Distribution import Distribution
from Gaussian import Gaussian
from numpy import zeros,ones,log,exp,array,vstack,hstack,sum,mean,kron,isnan
from numpy.random.mtrand import multinomial
from natter.DataModule import Data
from numpy.random import shuffle
import sys
from scipy import optimize
from functions import minimize_carl

class FiniteMixtureDistribution(Distribution):
    """
    Base class for a finite mixture of base distributions.

    .. math::
        p(x|\theta) = \sum_{k=1}^{K} \alpha_k p(x|\theta_k)

    """

    def __init__(self,
                 numberOfMixtureComponents=1,
                 baseDistribution=None):
        """
                
        :param numberOfMixtureComponents: Specifies the number of mixture components.
        :type numberOfMixtureComponents: int
        :param baseDistribution: each mixture component is of this type.
        :type baseDistribution: natter.Distributions.Distribution

        """
        self.numberOfMixtureComponents = numberOfMixtureComponents
        self.alphas = ones(self.numberOfMixtureComponents)/self.numberOfMixtureComponents
        self.ps = zeros(self.numberOfMixtureComponents,dtype=object)
        if baseDistribution==None:
            print "Warning: no base distribution specified! Assuming 2d Normal."
            baseDistribution = Gaussian()
        for k in xrange(self.numberOfMixtureComponents):
            self.ps[k] = baseDistribution.copy()
        self.dimensionality = baseDistribution.param['n']
        self.name = 'mixture of ' + baseDistribution.name
        self.primary = ['alpha','theta']


    def sample(self,m):
        """
        Samples from the mixture distribution.
        """
        nc = multinomial(m,self.alphas)
        mrange = range(m)
        shuffle(mrange)
        X = zeros((self.dimensionality,m))
        ind = 0
        for k in xrange(self.numberOfMixtureComponents):
            dat = self.ps[k].sample(nc[k])
            X[:,mrange[ind:ind + nc[k]]] = dat.X
            ind += nc[k]
        return Data(X,str(m) + " samples from a " + str(self.dimensionality) + self.name)

    def loglik(self,dat):
        X = self.ps[0].pdf(dat)*self.alphas[0]
        for k in xrange(1,self.numberOfMixtureComponents):
            X = X + self.ps[k].pdf(dat)*self.alphas[k]
        return log(X)

    def pdf(self,dat):
        return exp(self.loglik(dat))

    def primary2array(self):
        ret = array([])
        if 'alpha' in self.primary:
            ret = hstack((ret,self.alphas))
        if 'theta' in self.primary:
            for k in xrange(self.numberOfMixtureComponents):
                ret = hstack((ret,self.ps[k].primary2array()))
        return ret

    def array2primary(self,arr):
        if 'alpha' in self.primary:
            self.alphas = arr[0:self.numberOfMixtureComponents]
            arr = arr[self.numberOfMixtureComponents::]
        if 'theta' in self.primary:
            lp = len(self.ps[0].primary2array())
            for k in xrange(self.numberOfMixtureComponents):
                self.ps[k].array2primary(arr[0:lp])
                arr = arr[lp::]

    def dldtheta(self,dat):
        p = self.pdf(dat)
        n,m = dat.size()
        if 'alpha' in self.primary:
            ret=zeros((self.numberOfMixtureComponents,m))
            for k in xrange(self.numberOfMixtureComponents):
                ret[k,:] = self.ps[k].pdf(dat)/p
        if 'theta' in self.primary:
            ret0 = self.ps[0].dldtheta(dat)
            p0   = self.ps[0].pdf(dat)
            ret0 = ret0*(p0/p)
            for k in xrange(1,self.numberOfMixtureComponents):
                ret1 = self.ps[k].dldtheta(dat)
                p1   = self.ps[k].pdf(dat)
                ret1 = ret1*(p1/p)
                ret0 = vstack((ret0,ret1))
            if len(ret)==0:
                ret = ret0
            else:
                ret = vstack((ret,ret0))
        return ret

    def estimate(self,dat,verbose=False,method=None):
        """
        try to estimate the parameters using standard EM algorithm.
        
        """
        if method==None:
            method = 'gradient'
            print "Method: ", method
        n,m = dat.size()
        K   = self.numberOfMixtureComponents
        T = zeros((K,m))
        LP = zeros((K,m))
        nsamples = min([max([m,1000000]),10000000])
        Y = zeros((n,nsamples))
        for k in xrange(K):
            LP[k,:] = self.ps[k].loglik(dat)  + log(self.alphas[k])
        for k in xrange(K):
            T[k,:] = exp(LP[k,:])/sum(exp(LP),axis=0)
            
        def estep():
            if verbose:
                print "\rE",
                sys.stdout.flush()
            for k in xrange(K):
                LP[k,:] = self.ps[k].loglik(dat)  + log(self.alphas[k])
            for k in xrange(K):
                T[k,:] = exp(LP[k,:])/sum(exp(LP),axis=0)
            if verbose:
                print ".",
                sys.stdout.flush()
            return sum((T*LP).flatten())
        
        def mstep():
            L = T.copy()
            n,m = dat.size()
            if method=='sampling':
                for k in xrange(self.numberOfMixtureComponents):
                    ind =0
                    pdata = L[k,:]/sum(L[k,:])
                    MJ = multinomial(nsamples,pdata)
                    for nd in xrange(m):
                        if MJ[nd]>0:
                            Y[:,ind:ind+MJ[nd]] = kron(dat.X[:,nd],ones((MJ[nd],1))).T
                            ind = ind+MJ[nd]
                    self.ps[k].estimate(Data(Y))
                if 'alpha' in self.primary:
                    self.alphas = mean(T,axis=1)
            else:
                def f(ar):
                    par = ar.copy()
                    L = T.copy()
                    mg = len(self.ps[0].primary2array())
                    for k in xrange(self.numberOfMixtureComponents):
                        self.ps[k].array2primary(par[0:mg])
                        par = par[mg::]
                        X = self.ps[k].loglik(dat)
                        L[k,:] = T[k,:]*X
                    return -sum(L.flatten())
                def df(ar):
                    par = ar.copy()
                    grad = zeros(len(ar))
                    ind =0
                    mg = len(self.ps[0].primary2array())
                    for k in xrange(self.numberOfMixtureComponents):
                        self.ps[k].array2primary(par[0:mg])
                        par = par[mg::]
                        dX = self.ps[k].dldtheta(dat)
                        grad[ind:ind+mg] = sum(T[k,:]*dX,axis=1)
                        ind = ind+mg
                    return -grad
                def fdf(ar):
                    fv = f(ar)
                    grad = df(ar)
                    return fv,grad
                def check(arr):
                    err = optimize.check_grad(f,df,arr)
                    print "Error in gradient: ",err
                arr = self.primary2array()
                if 'alpha' in self.primary:
                    arr = arr[self.numberOfMixtureComponents::]
                check(arr)
                a = minimize_carl(arr,fdf,-10000)
                #                a = optimize.fmin_bfgs(f,arr,df,gtol=1e-09,disp=0,full_output=0,callback=check)
                if 'alpha' in self.primary:
                    self.alphas = mean(T,axis=1)

        diff = 10000000
        oldS = estep()
        while abs(diff)>1e-05:
            mstep()
            fv= estep()
            diff = oldS-fv
            oldS=fv
            print "Diff: ",diff
        
                
        

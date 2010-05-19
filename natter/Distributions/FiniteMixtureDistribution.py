from Distribution import Distribution
from Gaussian import Gaussian
from numpy import zeros,ones,log,exp,array,vstack,hstack,sum,mean,kron,isnan
from numpy.random.mtrand import multinomial
from natter.DataModule import Data
from numpy.random import shuffle
import sys
from scipy import optimize
#from functions import minimize_carl
from natter.Auxiliary.Numerics import logsumexp

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
        self.checkAlpha()
        n,m = dat.size()
        X = zeros((m,self.numberOfMixtureComponents))
        for k,p in enumerate(self.ps):
            X[:,k] = p.loglik(dat) + log(self.alphas[k])
        return logsumexp(X,axis=1)

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

    def checkAlpha(self):
        s=0.0
        # for alpha in self.alphas:
        #     ab = alpha
        #     alpha = max(min(alpha,1.0-s),0.0)
        #     s +=alpha
        #     diff =abs(ab-alpha)
        #     if diff > 1e-03:
        #         print "Warning : diff in alphas: ", diff
            
    def dldtheta(self,dat):
        self.checkAlpha()
        #        p = self.pdf(dat)
        lp = self.loglik(dat)
        n,m = dat.size()
        if 'alpha' in self.primary:
            ret=zeros((self.numberOfMixtureComponents,m))
            for k in xrange(self.numberOfMixtureComponents):
                ret[k,:] = exp(self.ps[k].loglik(dat)-lp)
        if 'theta' in self.primary:
            ret0 = self.ps[0].dldtheta(dat)
            lp0  = self.ps[0].loglik(dat)
            #p0   = self.ps[0].pdf(dat)
            ret0 = ret0*exp(lp0-lp + log(self.alphas[0]))
            for k in xrange(1,self.numberOfMixtureComponents):
                ret1 = self.ps[k].dldtheta(dat)
                lp1  = self.ps[k].loglik(dat)
                #                p1   = self.ps[k].pdf(dat)
                ret1 = ret1*exp(lp1-lp + log(self.alphas[k]))
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
            print "Using:  ", method , " - method"

        if method=="EMgradient" or method=="EMsampling":
            n,m = dat.size()
            K   = self.numberOfMixtureComponents
            T = zeros((K,m))
            LP = zeros((K,m))
            nsamples = min([max([m,1000000]),10000000])
            Y = zeros((n,nsamples))
            for k in xrange(K):
                LP[k,:] = self.ps[k].loglik(dat)  + log(self.alphas[k])
            for k in xrange(K):
                T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))

            def estep():
                if verbose:
                    print "\rE",
                    sys.stdout.flush()
                for k in xrange(K):
                    LP[k,:] = self.ps[k].loglik(dat)  + log(self.alphas[k])
                for k in xrange(K):
                    T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))
                if verbose:
                    print ".",
                    sys.stdout.flush()
                return sum((T*LP).flatten())

            def mstep():
                L = T.copy()
                n,m = dat.size()
                if method=='EMsampling':
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
                    a = optimize.fmin_bfgs(f,arr,df,gtol=1e-09,disp=0,full_output=0,callback=check)
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
        else:
            costf = 100000.0
            n,m = dat.size()
            def f(arr):
                self.array2primary(arr)
                # loss = zeros(self.numberOfMixtureComponents)
                # for k in xrange(self.numberOfMixtureComponents):
                #     loss[k] =max(exp(self.alphas[k]-1) -1,0) + max(exp(-self.alphas[k]) -1,0)
                # loss = costf*(sum(loss) +( exp(sum(self.alphas) -1)))
                # print "Loss:",loss
                LL=-sum(self.loglik(dat))
                ALL=-LL/(n*m)/log(2)  
                print "\rALL:", ALL
                return LL
            
            def df(arr):
                self.array2primary(arr)
                grad = sum(self.dldtheta(dat),axis=1)
                if 'alpha' in self.primary:
                    # for k,alpha  in enumerate(self.alphas):
                    #     if alpha>1.:
                    #         grad[k] = grad[k] - costf*exp(alpha -1)
                    #     elif alpha<0:
                    #         grad[k] = grad[k] - costf*exp(-alpha)
                    #     grad[k] = grad[k] - costf*exp(sum(self.alphas)-1)
                    nc = self.numberOfMixtureComponents
                    ga = grad[0:nc]
                    grad[0:nc] = ga - mean(ga)
                return -grad
            arr0 = self.primary2array()
            bound = [(None,None)]*len(arr0)
            for k in xrange(self.numberOfMixtureComponents):
                bound[k] = (0.,1.)
            arropt = optimize.fmin_l_bfgs_b(f,arr0,fprime=df,bounds=bound,pgtol=1e-13)[0]
            print "arropt : ",arropt
            #            arropt = optimize.fmin_l_bfgs_b(f,arropt,fprime=df,bounds=bound,pgtol=1e-10)[0]
            #            arropt = optimize.fmin_cg(f,arropt,df,gtol=1e-09)
        

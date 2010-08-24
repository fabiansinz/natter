from Distribution import Distribution
from Gaussian import Gaussian
from numpy import zeros,ones,log,exp,array,vstack,hstack,sum,mean,kron,isnan,dot
from numpy.random.mtrand import multinomial
from natter.DataModule import Data
from numpy.random import shuffle
import sys
from scipy import optimize
#from functions import minimize_carl
from copy import deepcopy

from natter.Auxiliary.Numerics import logsumexp

def logit(eta):
    return 1/(1+exp(-eta))

class FiniteMixtureDistribution(Distribution):
    """
    Base class for a finite mixture of base distributions.

    :math:`p(x|\\theta) = \sum_{k=1}^{K} \\alpha_k p(x|\\theta_k)`

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
        Distribution.__init__(self)

        self.param['numberOfMixtureComponents'] = numberOfMixtureComponents
        
        self.param['numberOfMixtureComponents'] = numberOfMixtureComponents
        self.param['alphas'] = ones(self.param['numberOfMixtureComponents'])/self.param['numberOfMixtureComponents']
        self.param['etas']   =  log(1/self.param['alphas'][0:-1] -1)
        self.param['ps'] = zeros(self.param['numberOfMixtureComponents'],dtype=object)
        if baseDistribution==None:
            print "Warning: no base distribution specified! Assuming 2d Normal."
            baseDistribution = Gaussian()
        for k in xrange(self.param['numberOfMixtureComponents']):
            self.param['ps'][k] = baseDistribution.copy()
        self.param['dimensionality'] = baseDistribution.param['n']
        self.name = 'mixture of ' + baseDistribution.name
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

    def __str__(self):
        s=  30*'-'
        s+=  '\n' + self.name + '\n'
        s+= "Number of Components: " + str( self.param['numberOfMixtureComponents']) + str('\n')
        for k,p in enumerate(self.param['ps']):
            s+= 30*'-'
            s+= "COMPONENT %d: "%k
            s+= 30*'-'
            s+= str(p)
        return s
 

    def sample(self,m):
        """
        Samples from the mixture distribution.
        """
        nc = multinomial(m,self.param['alphas'])
        mrange = range(m)
        shuffle(mrange)
        X = zeros((self.param['dimensionality'],m))
        ind = 0
        for k in xrange(self.param['numberOfMixtureComponents']):
            dat = self.param['ps'][k].sample(nc[k])
            X[:,mrange[ind:ind + nc[k]]] = dat.X
            ind += nc[k]
        return Data(X,str(m) + " samples from a " + str(self.param['dimensionality']) + self.name)

    def loglik(self,dat):
        self.checkAlpha()
        n,m = dat.size()
        X = zeros((m,self.param['numberOfMixtureComponents']))
        for k,p in enumerate(self.param['ps']):
            X[:,k] = p.loglik(dat) + log(self.param['alphas'][k])
        return logsumexp(X,axis=1)

    def pdf(self,dat):
        return exp(self.loglik(dat))

    def primary2array(self):
        ret = array([])
        if 'alpha' in self.primary:
            ret = hstack((ret,self.param['etas']))
            #            ret = hstack((ret,self.param['alphas']))
        if 'theta' in self.primary:
            for k in xrange(self.param['numberOfMixtureComponents']):
                ret = hstack((ret,self.param['ps'][k].primary2array()))
        return ret

    def array2primary(self,arr):
        if 'alpha' in self.primary:
            self.param['etas'] = arr[0:self.param['numberOfMixtureComponents']-1]
            self.param['alphas'][0:self.param['numberOfMixtureComponents']-1]= logit(self.param['etas'])
            self.param['alphas'][-1]=1-sum(self.param['alphas'][0:self.param['numberOfMixtureComponents']-1])
            #            self.param['alphas'] = arr[0:self.param['numberOfMixtureComponents']]
            arr = arr[self.param['numberOfMixtureComponents']-1::]
        if 'theta' in self.primary:
            lp = len(self.param['ps'][0].primary2array())
            for k in xrange(self.param['numberOfMixtureComponents']):
                self.param['ps'][k].array2primary(arr[0:lp])
                arr = arr[lp::]

    def checkAlpha(self):
        s=0.0
        # for alpha in self.param['alphas']:
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
        ret = array([])
        if 'alpha' in self.primary:
            ret=zeros((self.param['numberOfMixtureComponents']-1,m))
            dldp = zeros((self.param['numberOfMixtureComponents'],m))
            dpdeta = zeros((self.param['numberOfMixtureComponents']-1,self.param['numberOfMixtureComponents']))
            for k in xrange(self.param['numberOfMixtureComponents']-1):
                dldp[k,:] = exp(self.param['ps'][k].loglik(dat)-lp)
                dpdeta[k,k] = logit(self.param['etas'][k])*logit(-self.param['etas'][k])
            dldp[-1,:] = exp(self.param['ps'][-1].loglik(dat)-lp)
            dpdeta[:,-1]= -logit(self.param['etas'])*logit(-self.param['etas'])
            ret = dot(dpdeta,dldp)
        if 'theta' in self.primary:
            ret0 = self.param['ps'][0].dldtheta(dat)
            lp0  = self.param['ps'][0].loglik(dat)
            #p0   = self.param['ps'][0].pdf(dat)
            ret0 = ret0*exp(lp0-lp + log(self.param['alphas'][0]))
            for k in xrange(1,self.param['numberOfMixtureComponents']):
                ret1 = self.param['ps'][k].dldtheta(dat)
                lp1  = self.param['ps'][k].loglik(dat)
                #                p1   = self.param['ps'][k].pdf(dat)
                ret1 = ret1*exp(lp1-lp + log(self.param['alphas'][k]))
                ret0 = vstack((ret0,ret1))
            if len(ret)==0:
                ret = ret0
            else:
                ret = vstack((ret,ret0))
        return ret

    def estimate(self,dat,verbose=False,method=None):
        """
        try to estimate the parameters using standard EM algorithm.

        optionally, a method can be specified, possible values are:

        method='EMGradient'
           performs standard EM algorithm, the E-step
           is analytical, the M-step is done via gradient descent.

        method='Gradient':
           tries to directly perform a gradient ascent method on the (joint) likelihood.
           
        """
        if method==None:
            method = 'gradient'
            print "Using:  ", method , " - method"

        if method=="EMgradient" or method=="EMsampling":
            n,m = dat.size()
            K   = self.param['numberOfMixtureComponents']
            T = zeros((K,m))
            LP = zeros((K,m))
            nsamples = min([max([m,1000000]),10000000])
            Y = zeros((n,nsamples))
            for k in xrange(K):
                LP[k,:] = self.param['ps'][k].loglik(dat)  + log(self.param['alphas'][k])
            for k in xrange(K):
                T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))

            def estep():
                if verbose:
                    print "\rE",
                    sys.stdout.flush()
                for k in xrange(K):
                    LP[k,:] = self.param['ps'][k].loglik(dat)  + log(self.param['alphas'][k])
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
                    for k in xrange(self.param['numberOfMixtureComponents']):
                        ind =0
                        pdata = L[k,:]/sum(L[k,:])
                        MJ = multinomial(nsamples,pdata)
                        for nd in xrange(m):
                            if MJ[nd]>0:
                                Y[:,ind:ind+MJ[nd]] = kron(dat.X[:,nd],ones((MJ[nd],1))).T
                                ind = ind+MJ[nd]
                        self.param['ps'][k].estimate(Data(Y))
                    if 'alpha' in self.primary:
                        self.param['alphas'] = mean(T,axis=1)
                        self.param['etas']   = log(1/self.param['alphas'][0:self.param['numberOfMixtureComponents']-1] -1)
                else:
                    def f(ar):
                        par = ar.copy()
                        L = T.copy()
                        mg = len(self.param['ps'][0].primary2array())
                        for k in xrange(self.param['numberOfMixtureComponents']):
                            self.param['ps'][k].array2primary(par[0:mg])
                            par = par[mg::]
                            X = self.param['ps'][k].loglik(dat)
                            L[k,:] = T[k,:]*X
                        print "\rcurrent ALL : ", -mean(L.flatten())
                        return -sum(L.flatten())
                    def df(ar):
                        par = ar.copy()
                        grad = zeros(len(ar))
                        ind =0
                        mg = len(self.param['ps'][0].primary2array())
                        for k in xrange(self.param['numberOfMixtureComponents']):
                            self.param['ps'][k].array2primary(par[0:mg])
                            par = par[mg:]
                            dX = self.param['ps'][k].dldtheta(dat)
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
                        arr = arr[self.param['numberOfMixtureComponents']::]
                        #                    check(arr)
                    a = optimize.fmin_bfgs(f,arr,df,gtol=1e-09,disp=0,full_output=0)
                    if 'alpha' in self.primary:
                        self.param['alphas'] = mean(T,axis=1)
                        self.param['etas']   = log(1/self.param['alphas'][0:self.param['numberOfMixtureComponents']-1] -1)
            diff = 10000000
            oldS = estep()
            while abs(diff)>1e-05:
                mstep()
                fv= estep()
                diff = oldS-fv
                oldS=fv
                print "Diff: ",diff
        else:
            n,m = dat.size()
            
            def f(arr):
                self.array2primary(arr)
                LL=-sum(self.loglik(dat))
                print "\rnLL: ",LL,
                return LL
            
            def df(arr):
                self.array2primary(arr)
                grad = sum(self.dldtheta(dat),axis=1)
                # if 'alpha' in self.primary:
                #     nc = self.param['numberOfMixtureComponents']
                    # ga = grad[0:nc]
                    # grad[0:nc] = ga - mean(ga)
                return -grad
            arr0 = self.primary2array()
            # bound = [(None,None)]*len(self.param['alphas'])
            # for k in xrange(self.param['numberOfMixtureComponents']):
            #     bound[k] = (0.,1.)
            done=False
            diff = 100000;
            arr0 = self.primary2array()
            # print "Optimizing thetas...",
            # sys.stdout.flush()

            arropt = optimize.fmin_bfgs(f,arr0,fprime=df,maxiter=1000,gtol=1e-08) # be fast, be gr
            # oldALL = self.all(dat)
            # while not done:
            #     # self.primary = ['alpha']

            #     # print "Optimizing alphas...",
            #     # sys.stdout.flush()
            #     # n,m = dat.size()
            #     # K   = self.param['numberOfMixtureComponents']
            #     # T = zeros((K,m))
            #     # LP = zeros((K,m))
            #     # nsamples = min([max([m,1000000]),10000000])
            #     # Y = zeros((n,nsamples))
            #     # for k in xrange(K):
            #     #     LP[k,:] = self.param['ps'][k].loglik(dat)  + log(self.param['alphas'][k])
            #     # for k in xrange(K):
            #     #     T[k,:] = exp(LP[k,:]-logsumexp(LP,axis=0))
            #     # self.param['alphas'] = mean(T,axis=1)
            #     # arr0 = self.param['alphas']
            #     # #arropt = optimize.fmin_l_bfgs_b(f,arr0,fprime=df,bounds=bound,pgtol=1e-13)[0]
            #     # print "done."
            #     # print "alphas: ", self.param['alphas'], " sum(alphas): ", sum(self.param['alphas'])
            #     # print " Current ALL:",self.all(dat)
            #     # sys.stdout.flush() 

            #     # self.primary = ['theta']
            #     # arr0 = self.primary2array()
            #     # print "Optimizing thetas...",
            #     # sys.stdout.flush()

            #     arropt = optimize.fmin_bfgs(f,arr0,fprime=df,maxiter=10,gtol=1e-05) # be fast, be greedy
            #     # print "done."
            #     # sys.stdout.flush()
            #     # ALL = self.all(dat)
            #     # diff = abs(oldALL-ALL)
            #     # oldALL=ALL
            #     # if diff <1e-07:
            #     #     done=True
            #     # print "Current ALL: %g, current diff: %g" %(ALL,diff)
            #     # sys.stdout.flush()
        

from __future__ import division
from Distribution import Distribution
from numpy import zeros,eye,kron,dot,reshape,ones,log,pi,sum,diag,where,tril,hstack,squeeze,array,vstack,outer,exp,sqrt,abs
from numpy.random import randn,rand
from numpy.linalg import cholesky,inv,solve,svd
from natter.DataModule import Data
from scipy import optimize
from Gaussian import Gaussian
from natter.Auxiliary.Errors import InitializationError, NumericalExceptionError
from scipy.optimize import check_grad
from random import shuffle
import sys
from copy import deepcopy
import os.path
import pickle
from natter.Auxiliary.Numerics import logsumexp
import warnings



HaveParallelSupport=False               # does not work: parallel python sucks
try:
    import pp
    fname = "~/.ppservers"              # file containing the servers to which I allowed to connect to
    if os.path.isfile(fname):
        pslist = pickle.load(open(fname))
    else:
        pslist = ("localhost",)         # otherwise use localhost only
    moduleList = ('numpy', 'numpy.random','pickle','sys','copy','natter.Distributions.Gaussian','natter.Distributions.Distribution','natter.DataModule','natter.Auxiliary.Errors','scikits.openopt','scipy.optimize')
    GLOBALS = globals()
    job_server = pp.Server(ppservers=pslist) 
    HaveParallelSupport=True
except:
    pass
HaveParallelSupport=False
    

class GPPM(Distribution):
    """
    Gaussian Process Product model.
    The data is modeled by:

    :math:`y(x) = f(x) \\exp(h(x)) + \\epsilon`
    where f,h are Gaussian processes over the possibly two-dimensional pixel space.
    However, it is assumed, that the xs are the same for all data points.

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.


    :param param:
        dictionary which may contain parameters for the hidden GPs:
        'n' : dimensionality
        'f' : Gaussian distribution of dimensionality n 
        'h' : Gaussian distribution of dimensionality n 
    :type param: dict

    Primary parameters are ['f','h'], indicating the distributions of
    the hidden functions. For each such distribution, primary
    parameters can be specified, which are then selected to be
    inferred for estimation.
    """
    def __init__(self, *args,**kwargs):
        Distribution.__init__(self,param)
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
        if param==None:
            self.param['n'] = 16        # 4x4 images are default
            self.param['f'] = Gaussian({'n':16})
            self.param['f'].primary = ['mu','sigma']
            self.param['h'] = Gaussian({'n':16})
            self.param['h'].primary = ['mu','sigma']
            self.param['obsSig'] = 1.0  # observation noise, uniform, this is no
            self.param['lik']   = Gaussian({'n': 16,'sigma': eye(16)})
            
        else:
            for key in param.keys():
                self.param[key]=param[key]
            if 'n' not in self.param.keys():
                raise InitializationError('Parameter \'n\' must be specified')
            if 'f' not in self.param.keys():
                self.param['f'] = Gaussian({'n':self.param['n']})
                self.param['f'].primary = ['sigma']
            if 'h' not in self.param.keys():
                self.param['h'] = Gaussian({'n':self.param['n']})
                self.param['h'].primary = ['sigma']
            if 'obsSig' not in self.param.keys():
                self.param['obsSig']=1.0
        self.param['lik']   = Gaussian({'n': self.param['n'],
                                        'sigma': self.param['obsSig']*eye(self.param['n'])})
        self.primary = ['f','h']
        self.name = 'Gaussian Process Product Distribution'
        self.samplesUp2Date=False
        self.samples = {'f':array([[]]),
                        'h': {} }
        self.currentIndex=-1
                        
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


    def sample(self,m):
        """
        Samples m samples from the model. First sample the hidden
        processes and then multiply them and add observation noise to
        it.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples

        """

        F = self.param['f'].sample(m).X
        H = self.param['h'].sample(m).X
        N = randn(self.param['n'],m)*sqrt(self.param['obsSig'])
        return Data(F*exp(H) + N, str(m) + ' sapmles from a ' + self.name)


    

    def gibbsSampleSingle(self,data,i,nSamples=1,interimSamples=1):
        """
        Performs a Gibbs sampling sweep for the posterior distribution
        for the i-th datapoint.  First f conditioned on h,x can be
        jointly sampled from a Gaussian. For h conditioned on f,x, we
        perform a slice sampling procedure.  As slice sampling is a
        Markov chain, increasing the number of interimSamples might
        give indepenend samples but slows down sampling.

        :math:`p(f,h|x)`

        :param data: Data object containing the x at the i-th position
        :type data: DataModule.Data
        :param i: position of the datapoint (x) to condition on
        :type i: int
        :param nSamples: number of samples to draw (default:1)
        :type nSamples: int
        :param interimSamples: number of samples between two actual samples (default:1)
        :type interimSamples: int

        :returns: Dictionary with possibly two entries:
                  'f' : samples for the hidden function F
                  'h' : samples for the hidden function H
                  depending on which of the hidden distributions are in the primary field.
                  
        
        """
        if self.samplesUp2Date and self.samples['f'].shape[1]==nSamples and i==self.currentIndex:
            return
        self.currentIndex=i
        X = data.X[:,i].flatten()       # datapoint to condition on 
        pr = ['f','h']                  # we have to iterate between these two functions to sample from
        DG = deepcopy(self.param['f'])  # dummy Gauss to sample from
        samples = {}
        if 'f' in self.primary:
            samples['f'] = zeros((len(X),nSamples))
        if 'h' in self.primary:
            samples['h'] = zeros((len(X),nSamples))
        for k in range(nSamples):
            print "\r %d/%d" %(k,nSamples),
            for j in range(interimSamples):
                shuffle(pr)             # random order of primary parameters
                for p in pr:
                    if p=='f':              # sample from the conditional Gaussian
                        H = self.stateCache['h']['lastSample'][:,i]
                        LI = cholesky(diag((exp(2*H)/self.param['obsSig'])) + dot(self.param['f'].cholP,self.param['f'].cholP.T))
                        DG.cholP= LI
                        F = DG.sample(1).X[:,0] # sample one F|X,H
                        self.stateCache['f'][:,i]=F
                    if p=='h':                  # for h we do slice sampling
                        F = self.stateCache['f'][:,i] # last sample
                        H = self.stateCache['h']['lastSample'][:,i]
                        I = range(len(X))
                        shuffle(I)      # random order for the H-components
                        for l in I:
                            y = self.jointLogLik(X,F,H) # log-height at current point
                            y = log(rand(1)*exp(y))     # sample new log-heigth slice
                            def fh(h):
                                H[l]=h
                                return [self.jointLogLik(X,F,H)-y]
                            def dfh(h): # gradient of the joint log-likelihood wrt H[l]
                                H[l]=h
                                g  = (X[l]-F[l]*exp(H[l]))*exp(H[l])*F[l]
                                g += dot(self.param['h'].cholP,dot(self.param['h'].cholP.T,self.param['h'].param['mu']-H))[l]
                                return g
                            xl = H[l] - self.stateCache['h']['lastInterval'][l,i]
                            xr = H[l] + self.stateCache['h']['lastInterval'][l,i]
                            # try:
                            #     raise NumericalExceptionError('could not solve least squares problem')
                            #     ip = LSP(fh, xr, df=dfh, show=False,diffInt = 1.5e-8, xtol = 1.5e-8, ftol = 1.5e-8)
                            #     ip.iprint=-1 # silent mode
                            #     rr = ip.solve('nlp:ralg')
                            #     ip = LSP(fh, xl, df=dfh, show=False, diffInt = 1.5e-8, xtol = 1.5e-8, ftol = 1.5e-8)
                            #     ip.iprint=-1
                            #     rl = ip.solve('nlp:ralg')
                            #     if abs(rr.xf-rl.xf)>0.5: # found the right interval via optimization
                            #         xl = min(rr.xf,rl.xf)
                            #         xr = max(rr.xf,rl.xf)
                            #         self.stateCache['h']['lastInterval'][l,i]=(xr-xl)/2.0
                            #         H[l] = rand(1)*(xr-xl) + xl # uniform in (xl,xr)
                            #         self.stateCache['h']['lastSample'][l,i]=H[l]
                            #     else:
                            #         raise NumericalExceptionError('could not solve least squares problem')
                            # except TypeError:       # have to perform a step-out/shrink procedure
                            #     print "LSP crashed, doing standard slice sampling!"
                            # except NumericalExceptionError:
                            #     pass    # nothing to be done here
                            # else:
                                
                            r = rand(1)
                            w=self.stateCache['h']['lastInterval'][l,i]
                            w=0.7
                            xl = H[l] -     r*w
                            xr = H[l] + (1-r)*w
                            while fh(xl)[0]>0:
                                xl -=w
                            while fh(xr)[0]>0:
                                xr +=w
                            self.stateCache['h']['lastInterval'][l,i]=(xr-xl)/2
                            reject = True
                            while reject:
                                hinew = rand(1)*(xr-xl) + xl # uniform in (xl,xr)
                                reject = fh(hinew)[0]<0
                            self.stateCache['h']['lastSample'][l,i]=H[l]
            samples['f'][:,k]=self.stateCache['f'][:,i]
            samples['h'][:,k]=self.stateCache['h']['lastSample'][:,i]
        self.samples = samples
        self.samplesUp2Date = True
    
            
                                

    def primary2array(self):
        ret = array([])
        if 'f' in self.primary:
            ret = hstack((ret,self.param['f'].primary2array()))
        if 'h' in self.primary:
            ret = hstack((ret,self.param['h'].primary2array()))
        if 'obsSig' in self.primary:
            ret = hstack((ret,array([self.param['obsSig']])))
        return ret


    def array2primary(self,arr):
        if 'f' in self.primary:
            szf = len(self.param['f'].primary2array())
            self.param['f'].array2primary(arr[0:szf])
            arr = arr[szf::]
        if 'g' in self.primary:
            szf = len(self.param['g'].primary2array())
            self.param['g'].array2primary(arr[0:szf])
            arr = arr[szf::]
        if 'h' in self.primary:
            szf = len(self.param['h'].primary2array())
            self.param['h'].array2primary(arr[0:szf])
            arr = arr[szf::]
        if 'obsSig' in self.primary:
            self.param['obsSig']=arr[-1]
            self.param['lik'].param['sigma'] = eye(self.param['n'])*self.param['obsSig']
        self.samplesUp2Date=False


    def jointLogLik(self,X,F,H):
        """
        Computes the joint log-likelihood for a tripplet X,F,H

        :param X: Datapoint
        :type X:  numpy.array
        :param F: Hidden function(vector) f
        :type F: numpy.array
        :param H: Hidden function (vector) h
        :type H: numpy.array

        :returns: log-likelihood value, depending on the observation noise 'obsSig'
        :math:`p(x,f,h)=\\mathcal{N}(x|f \\exp(h), \\sigma) \\mathcal{N}(f|\\mu_f, \\Sigma_f) \\mathcal{N}(h|\\mu_h,\\Sigma_h)`


        """
        MUO = F*exp(H)
        self.param['lik'].param['mu'] = MUO
        LL= self.param['lik'].loglik(Data(X.reshape((len(F),1))))
        LL = LL + self.param['f'].loglik(Data(F.reshape((len(F),1))))
        LL = LL + self.param['h'].loglik(Data(H.reshape((len(F),1))))
        return LL
    



    




    def gradJointLogLikAll(self,data,F,G,H):
        n,m = data.size()
        grad = zeros(len(F)*3)
        grad[0:len(F)] = sum(data.X - kron(F*G*exp(H),ones(m,1)).T,axis=1)*(G*exp(H))/self.param['obsSig']
        grad[len(F):2*len(F)] = sum(data.X - kron(F*G*exp(H),ones(m,1)).T,axis=1)*(F*exp(H))/self.param['obsSig']
        grad[2*len(F):3*len(F)] = sum(data.X - kron(F*G*exp(H),ones(m,1)).T,axis=1)*(G*F*exp(H))/self.param['obsSig']
        prim = self.param['f'].primary
        self.param['f'].primary=['mu']
        grad[0:len(F)] = grad[0:len(F)] - sum(self.param['f'].dldtheta(Data(kron(F,ones(m,1)).T)),axi=1)
        self.param['f'].primary=prim
        prim = self.param['g'].primary
        self.param['g'].primary=['mu']
        grad[len(F):2*len(F)] = grad[len(F):2*len(F)] - sum(self.param['g'].dldtheta(Data(kron(G,ones(m,1)).T)),axi=1)
        self.param['g'].primary=prim
        prim = self.param['h'].primary
        self.param['h'].primary=['mu']
        grad[2*len(F):3*len(F)] = grad[2*len(F):3*len(F)] - sum(self.param['h'].dldtheta(Data(kron(H,ones(m,1)).T)),axi=1)
        self.param['h'].primary=prim
        return grad


    def gradJointLogLik(self,X,F,H):
        """
        Gradient of the joint log-likelihood with respect to the
        hidden functions. evaluated at the given hidden functions.

        :param X: datapoint
        :type  X: numpy.array
        :param F: hidden F
        :tpye  F: numpy.array
        :param H: hidden H
        :type  H: numpy.array
        
        
        """
        grad = zeros(3*len(F))
        grad[0:len(F)] = (X - F*G*exp(H))*(G*exp(H))/self.param['obsSig']
        grad[len(F):2*len(F)] = (X - F*G*exp(H))*(F*exp(H))/self.param['obsSig']
        grad[2*len(F):3*len(F)] = (X - F*G*exp(H))*(G*F*exp(H))/self.param['obsSig']
        prim = self.param['f'].primary
        self.param['f'].primary=['mu']
        grad[0:len(F)] = grad[0:len(F)] - self.param['f'].dldtheta(Data(F.reshape((len(F),1)))).flatten()
        self.param['f'].primary=prim
        prim = self.param['g'].primary
        self.param['g'].primary=['mu']
        grad[len(F):2*len(F)] = grad[len(F):2*len(F)] - self.param['g'].dldtheta(Data(G.reshape((len(F),1)))).flatten()
        self.param['g'].primary=prim
        prim = self.param['h'].primary
        self.param['h'].primary=['mu']
        grad[2*len(F):3*len(F)] = grad[2*len(F):3*len(F)] - self.param['h'].dldtheta(Data(H.reshape((len(F),1)))).flatten()
        self.param['h'].primary=prim
        return grad
    
        
    def hessianJointAll(self,data,F,G,H):
        sz = len(F)
        n,m=data.size()
        CF = solve(self.param['f'].cholP,solve(self.param['f'].cholP.T,eye(sz)))
        CG = solve(self.param['g'].cholP,solve(self.param['g'].cholP.T,eye(sz)))
        CH = solve(self.param['h'].cholP,solve(self.param['h'].cholP.T,eye(sz)))
        hes = zeros((3*sz,3*sz))
        hes[0:sz,0:sz] = -m*CF - m*diag(G**2*exp(2*H))
        hes[sz:2*sz,sz:2*sz] = -m*CG - m*diag(F**2*exp(2*H))
        hes[2*sz:3*sz,2*sz:3*sz] = -m*CH + diag(sum(data.X - kron(F*G*exp(H),ones(m,1)).T,axis=1) -2*F**2*G**2*exp(2*H)*H)
        return hes


    def hessianJoint(self,X,F,G,H):
        sz = len(F)
        CF = dot(self.param['f'].cholP,self.param['f'].cholP.T) # inverse covariance
        CG = dot(self.param['g'].cholP,self.param['g'].cholP.T)
        CH = dot(self.param['h'].cholP,self.param['h'].cholP.T)
        hes = zeros((3*sz,3*sz))
        # main diagonal
        hes[0:sz,0:sz] = -CF - diag((G**2)*exp(2*H)/self.param['obsSig'])
        hes[sz:2*sz,sz:2*sz] = -CG - diag((F**2)*exp(2*H)/self.param['obsSig'])
        hes[2*sz:3*sz,2*sz:3*sz] = -CH + diag((X*F*G*exp(H) -2*(F**2)*(G**2)*exp(2*H))/self.param['obsSig'])
        # f-g / g-f part
        hes[0:sz,sz:2*sz]=hes[sz:2*sz,0:sz] = diag((X*exp(H)- 2*F*G*exp(2*H))/self.param['obsSig'])
        # f-h/h-f part
        hes[0:sz,2*sz:3*sz]=hes[2*sz:3*sz,0:sz] = diag((X*exp(H)*G- 2*F*(G**2)*exp(2*H))/self.param['obsSig'])
        # g-h/h-g part
        hes[sz:2*sz,2*sz:3*sz]=hes[2*sz:3*sz,sz:2*sz] = diag((X*exp(H)*F- 2*G*(F**2)*exp(2*H))/self.param['obsSig'])
        return hes
        
        
    def loglik(self,data,method=None,nSamples=10):
        """
        Approximately calculates the likelihood.  The approximation
        can be done in tow possible ways: Laplace or MCMC.  For the
        laplace approximation a second order taylor expansion is
        integrated analytically. For MCMC we use a bunch of samples to
        approximate the likelihood.

        :param data: Data to calculate the log-liklihood on.
        :type data:  DataModule.Data
        :param method: Optional argument specifying the method to use for approximation. (Default is MCMC):
                       'Laplace' or 'MCMC' are possible values
        :type method: str
        :param nSamples: Optional argument to specify the number of samples used for each
                         datapoint to approximate the likelihood for this datapoint.
                         If the specified method is 'Laplace' this parameters has no effect.
        :type nSamples: int

        :returns: numpy.array containing the log-likelihood values for each datapoint.
        """

        LL = zeros(data.size()[1])
        if HaveParallelSupport:
            results=[]
            for k in range(len(LL)):    # submit all
                f = job_server.submit(self.loglikSingle,             # function
                                      (data,k,method,None,nSamples), # arguments
                                      modules=("numpy",),
                                      globals=GLOBALS)
                results.append([k,f])
            for i,f in results:         # collect the results
                LL[i]=f()
        else:
            for k in range(len(LL)):
                LL[k]=self.loglikSingle(data,k,method=method,nSamples=nSamples)
        return LL


    def getMAP(self,data,i):
        X = data.X[:,i]
        Y = self.stateCache[:,i]
        YI = Y.copy()
        if 'f' in self.primary:
            FI = YI[0:len(X)]
            YI = YI[len(X)::]
        if 'g' in self.primary:
            GI = YI[0:len(X)]
            YI=Y[len(X)::]
        if 'h' in self.primary:
            HI = YI[0:len(X)]
        def f(Y):
            if 'f' in self.primary:
                FU = Y[0:len(X)]
            else:
                FU = ones(len(X))
            if 'g' in self.primary:
                GU = Y[len(X):2*len(X)]
            else:
                GU = ones(len(X))
            if 'h' in self.primary:
                HU = Y[2*len(X):3*len(X)]
            else:
                HU = zeros(len(X))
            return -self.jointLogLik(X,FU,GU,HU)
        def df(Y):
            if 'f' in self.primary:
                FU = Y[0:len(X)]
            else:
                FU = ones(len(X))
            if 'g' in self.primary:
                GU = Y[len(X):2*len(X)]
            else:
                GU = ones(len(X))
            if 'h' in self.primary:
                HU = Y[2*len(X):3*len(X)]
            else:
                HU = zeros(len(X))
            return -self.gradJointLogLik(X,FU,GU,HU)
        def callb(theta):               # for debugging only
            err = check_grad(f,df,theta)
            print "Error in gradient: " ,err
        Y = optimize.fmin_bfgs(f,Y,df,disp=0,maxiter=5)
        self.stateCache[:,i]=Y
        if 'f' in self.primary:
            F = Y[0:len(X)]
        else:
            F=ones(len(X))
        if 'g' in self.primary:
            G = Y[len(FI):2*len(X)]
        else:
            G=ones(len(X))
        if 'h' in self.primary:
            H = Y[2*len(X):3*len(X)]
        else:
            H=zeros(len(FI))
        return F,G,H


    def dldtheta(self,data,method=None,nSamples=10):
        """
        Approximately calculates the gradien of the log- likelihood.  The approximation
        can be done in tow possible ways: Laplace or MCMC.  For the
        laplace approximation a second order taylor expansion is
        integrated analytically and then the derivative is calculated.
        For MCMC we use a bunch of samples to   approximate the likelihood.

        :param data: Data to calculate the log-liklihood on.
        :type data:  DataModule.Data
        :param method: Optional argument specifying the method to use for approximation. (Default is MCMC):
                       'Laplace' or 'MCMC' are possible values
        :type method: str
        :param nSamples: Optional argument to specify the number of samples used for each
                         datapoint to approximate the likelihood for this datapoint.
                         If the specified method is 'Laplace' this parameters has no effect.
        :type nSamples: int

        :returns: numpy.array containing the log-likelihood values for each datapoint.
        """

        n,m=data.size()
        ndf = len(self.primary2array())
        grad = zeros((ndf,m))
        if HaveParallelSupport:
            results=[]
            for k in range(m):    # submit all
                f = job_server.submit(self.dldthetaSingle,
                                      (data,k,method,None,nSamples),
                                      modules=("numpy",),
                                      globals=GLOBALS)
                results.append([k,f])
            for i,f in results:         # collect the results
                grad[:,i]=f()
        else:
            for k in range(m):
                grad[:,k]= self.dldthetaSingle(data,k,method=method,nSamples=nSamples)        
        # pf = self.param['f'].primary2array()
        # ph = self.param['h'].primary2array()
        # gn = len(pf)+len(pg)+len(ph)
        # grad = zeros((gn,m))
        # for i,x in enumerate(data.X.T):
        #     grad[:,i] = self.dldthetaSingle(data,i,method=method)
        return grad
        

    def dldthetaSingle(self,data,i,method=None,nSamples=10):
        """
        
        Computes the gradient of the (approximate) likelihood for a single datapoint.

        :param data: data 
        :type data: DataModule.Data
        :param i: index of the datapoint to calculate the gradient
        :type i: int
        :param samples: optional argument to pass samples which are drawn from :math:`p(f,h|x)`.
        :type samples:  dict (same format as the returned from gibbsSampleSingle)
        :param nSamples: if no set of samples is passed, a new set is generated with as many samples as nSamples
        :type nSamples: int

        :returns: an numpy.array containing the approximate gradient for the single datapoint.

        """
        if method==None:
            method = 'Laplace'
        if method =='Laplace':
            warnings.warn("This method is not functional yet!", UserWarning)
            gn=0                            # numper of parameters
            if 'f' in self.primary:
                pf = self.param['f'].primary2array()
                nf = self.param['f'].param['n']
                gn = gn+len(pf)
                indF = array(range(len(pf)))
            if 'g' in self.primary:
                pg = self.param['g'].primary2array()
                ng = self.param['g'].param['n']
                indG = array(range(len(pg)))+gn
                gn = gn+len(pf)
            if 'h' in self.primary:
                ph = self.param['h'].primary2array()
                nh = self.param['h'].param['n']
                indH = array(range(len(ph)) ) + gn
                gn = gn + len(ph)
            grad = zeros(gn)

            # get the maximal likely hidden functions, if neccesary

            F,G,H = self.getMAP(data,i)

            # approximation to the covariance/hessian
            nhes  = -self.hessianJoint(data.X[:,i],F,G,H)
            iHess = inv(nhes+ eye(nhes.shape[0])*1e-05)

            # gradient of the first term in the laplace approximation 
            if 'f' in self.primary:
                grad[indF] = self.param['f'].dldtheta(Data(F.reshape((len(F),1)))).flatten()
            if 'g' in self.primary:
                grad[indG] = self.param['g'].dldtheta(Data(G.reshape((len(F),1)))).flatten()
            if 'h' in self.primary:
                grad[indH] = self.param['h'].dldtheta(Data(H.reshape((len(F),1)))).flatten()

            # now gradient of the log-det of the hessian. 
            o =0
            oh = 0
            if 'f' in self.primary:         # the indexing only works, if sigma are the only primary parameters for the hidden factors
                grad[indF]-= (dot(iHess[oh:oh+nf,oh:oh+nf],self.param['f'].cholP))[self.param['f'].I]
                o = o+len(pf)
                oh = oh+nf
            if 'g' in self.primary:
                grad[indG]-= (dot(iHess[oh:oh+ng,oh:oh+ng],self.param['g'].cholP))[self.param['g'].I]
                o=o+len(pg)
                oh+=ng
            if 'h' in self.primary:
                grad[indH]-= (dot(iHess[oh:oh+nh,oh:oh+nh],self.param['h'].cholP))[self.param['h'].I]
        else:
            # do MCMC stuff
            if not self.samplesUp2Date:
                self.gibbsSampleSingle(data,i,nSamples=nSamples)
            else:
                nSamples = self.samples['f'].shape[1]
            LLS = zeros(nSamples)       # log likelihood values
            ndf = len(self.primary2array())
            DFS = zeros((ndf,nSamples)) # gradient of log likelihoods
            nf = len(self.param['f'].primary2array())
            nh = len(self.param['h'].primary2array())
            for k in range(nSamples):
                LLS[k] = self.jointLogLik(data.X[:,i],self.samples['f'][:,k],self.samples['h'][:,k])
                if 'f' in self.primary:
                    gf = self.param['f'].dldtheta(Data(self.samples['f'][:,k].reshape((self.param['n'],1)))).flatten()
                if 'h' in self.primary:
                    gh = self.param['h'].dldtheta(Data(self.samples['h'][:,k].reshape((self.param['n'],1)))).flatten()
                DFS[:,k]= hstack((gf,gh))*LLS[k]
            grad = logsumexp(DFS,axis=1)-logsumexp(LLS)
        return grad
          
        
    def loglikSingle(self,data,i,method=None,nSamples=10):
        """
        Computes the (approximate) log-likelihood for a single
        datapoint via MCMC or via Laplace approximation.

        :param data: all data
        :type data:  DataModule.Data
        :param i:    index of the datapoint to evaluate the likelihood
        :type i:     int
        :param method: optional argument to specify the method to use for the approximation
        :type method: string
        :param samples: optional argument to pass a set of samples, usually obtained from the function gibbsSampleSingle.
        :type samples: dict, containing the keys 'f' and 'h' each having the dimensions (n,m) where n is the dimensionality of the datapoint and m are the number of samples
        :param nSamples: if no set of samples is passed, then nSamples new samples are drawn.
        :type nSamples: int

        :returns: approximate value of the log-likelihood function.
                
        """
        if method == None:
            method = 'Laplace'
        if method=='Laplace':        # ATTENTION: Does not work right now
            # maximal likely hidden functions
            F,G,H = self.getMAP(data,i)

            # negative (inverse) hessian at that point as approximate covariance
            nhes = -self.hessianJoint(data.X[:,i],F,G,H) 

            # for log-det of the hessian/covariance, we add a ridge to avoid numerical instability
            s = svd(nhes + eye(nhes.shape[0])*1e-05,compute_uv=False)

            # and now use the laplace approximation 
            LL = self.jointLogLik(data.X[:,i],F,G,H) + 0.5*log(2*pi)*3*len(F) - 0.5*sum((log(s)))
        else:
            if not self.samplesUp2Date:
                self.gibbsSampleSingle(data,i,nSamples=nSamples)
            else:
                nSamples = self.samples['f'].shape[1]
            LLS = zeros(nSamples)
            for k in range(nSamples):
                LLS[k] = self.jointLogLik(data.X[:,i],self.samples['f'][:,k],self.samples['h'][:,k])
            LL = logsumexp(LLS)
        return LL
        


    def initStateCache(self,data):
        """
        Initialize the internal cache for remembering either MAPs or
        last samples for Gibbs sampling to speed up computations. It
        depends on the data, as there need to be as many remembered
        MAPs as there are datapoints.

        :param data: Data to be used for inference.
        :type data: DataModule.Data
        
        """
        n,m = data.size()
        self.stateCache = {}
        self.stateCache['f'] = ones((n,m))
        self.stateCache['h'] = {'lastSample': zeros((n,m)),
                                    'lastInterval': ones((n,m))}

        
    def estimate(self,data,method=None):
        """
        Estimates the parameters of the underlying processes.  By
        default this is done by using the Laplace approximation in the
        same way as in an Olshausen and Field model.
        
        :param data: Data points on which the GPPM model will be estimated.
        :type data: natter.DataModule.Data

        :param method: Optional argument specifies the method to fit the data
        :type method: string
        
        """
        self.initStateCache(data)
        if method == None:
            method = 'MCMC'
        def f(theta):
            self.array2primary(theta)
            print ".",
            sys.stdout.flush()
            LL = self.loglik(data,method=method)
            return -sum(LL)
        def df(theta):
            self.array2primary(theta)
            grad = -self.dldtheta(data,method=method)
            return sum(grad,axis=1)
        def callb(theta):
            err = check_grad(f,df,theta)
            print "Error in gradient: " ,err
            sys.stdout.flush()
        theta0=self.primary2array()
        optimize.fmin_bfgs(f,theta0,df,callback=callb)

            

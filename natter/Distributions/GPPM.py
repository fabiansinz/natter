from __future__ import division
from Distribution import Distribution
from numpy import zeros,eye,kron,dot,reshape,ones,log,pi,sum,diag,where,tril,hstack,squeeze,array,vstack,outer,exp,sqrt,abs
from numpy.random import randn,rand
from numpy.linalg import cholesky,inv,solve,svd
from natter.DataModule import Data
from scipy import optimize
from Gaussian import Gaussian
from natter.Auxiliary.Errors import InitializationError
from scipy.optimize import check_grad
from scikits.openopt import LSP
from random import shuffle
import sys
from copy import deepcopy

class GPPM(Distribution):
    """
    Gaussian Process Product model.
    The data is modeled by:

    :math:`y(x) = f(x) g(x) \\exp(h(x)) + \\epsilon`
    where f,g,h are Gaussian processes over the possibly two-dimensional pixel space.
    However, it is assumed, that the xs are the same for all data points.

    :param param:
        dictionary which may contain parameters for the hidden GPs:
        'n' : dimensionality
        'f' : Gaussian distribution of dimensionality n 
        'g' : Gaussian distribution of dimensionality n 
        'h' : Gaussian distribution of dimensionality n 
    :type param: dict

    Primary parameters are ['f','g','h']
    """
    def __init__(self,param=None):
        Distribution.__init__(self,param)
        if param==None:
            self.param['n'] = 16        # 4x4 images are default
            self.param['f'] = Gaussian({'n':16})
            self.param['f'].primary = ['sigma']
            self.param['g'] = Gaussian({'n':16})
            self.param['g'].primary = ['sigma']
            self.param['h'] = Gaussian({'n':16})
            self.param['h'].primary = ['sigma']
            self.param['obsSig'] = 1.0  # observation noise
        else:
            for key in param.keys():
                self.param[key]=param[key]
            if 'n' not in self.param.keys():
                raise InitializationError('Parameter \'n\' must be specified')
            if 'f' not in self.param.keys():
                self.param['f'] = Gaussian({'n':self.param['n']})
                self.param['f'].primary = ['sigma']
            if 'g' not in self.param.keys():
                self.param['g'] = Gaussian({'n':self.param['n']})
                self.param['g'].primary = ['sigma']
            if 'h' not in self.param.keys():
                self.param['h'] = Gaussian({'n':self.param['n']})
                self.param['h'].primary = ['sigma']
            if 'obsSig' not in self.param.keys():
                self.param['obsSig']=1.0
        self.primary = ['f','g','h']
        self.name = 'Gaussian Process Product Distribution'
        self.param['lik'] = Gaussian({'n':self.param['n'],
                                      'sigma':eye(self.param['n'])*self.param['obsSig']})
        
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
        G = self.param['g'].sample(m).X
        H = self.param['h'].sample(m).X
        N = randn(self.param['n'],m)*sqrt(self.param['obsSig'])
        return Data(F*G*exp(H) + N, str(m) + ' sapmles from a ' + self.name)



    def gibbsSampleSingle(self,data,i,nSamples=1):
        """
        Performs a Gibbs sampling sweep for the posterior distribution for the i-th datapoint.
        :math:`p(f,g,h|x)`
        
        """
        Y = self.stateCache[:,i].flatten()        # last sample for this datapoint
        X = data.X[:,i].flatten()
        pp = deepcopy(self.primary)
        DG = deepcopy(self.param['f'])     # dummy Gauss to sample from
        F = Y[0:len(X)]
        G = Y[len(X):2*len(X)]
        H = Y[2*len(X):3*len(X)]
        S = zeros((len(Y),nSamples))
        for k in range(nSamples):
            shuffle(pp)
            for p in pp:
                G[where(abs(G)<1e-05)] = 1e-05*ones(len(X))[where(abs(G)<1e-05)]
                F[where(abs(F)<1e-05)] = 1e-05*ones(len(X))[where(abs(F)<1e-05)]
                if p=='f':              # sample from the conditional Gaussian
                    LI = cholesky(diag((G**2*exp(2*H)/self.param['obsSig'])) + dot(self.param['f'].cholP,self.param['f'].cholP.T))
                    DG.cholP= LI
                    F = DG.sample(1).X[:,0] # sample one F|X,G,H
                if p=='g':
                    LI = cholesky(diag((F**2*exp(2*H)/self.param['obsSig'])) + dot(self.param['g'].cholP,self.param['f'].cholP.T))
                    DG.cholP= LI
                    G = DG.sample(1).X[:,0] # sample one G|X,F,H
                if p=='h':                  # for h we do slice sampling
                    I = range(len(X))
                    shuffle(I)
                    for l in I:
                        y = self.jointLogLik(X,F,G,H) # log-height at current point
                        y = log(rand(1)*exp(y))       # sample new log-heigth slice
                        def fh(h):
                            H[l]=h
                            return [self.jointLogLik(X,F,G,H)-y]
                        def dfh(h):
                            H[l]=h
                            g  = (X[l]-F[l]*G[l]*exp(H[l]))*exp(H[l])*F[l]*G[l]
                            g += dot(self.param['h'].cholP,dot(self.param['h'].cholP.T,self.param['h'].param['mu']-H))[l]
                            return g
                        sg = 50./(1./self.param['h'].param['sigma'][l,l]+ (X[l]**2)*abs(2.-F[l]*G[l])/self.param['obsSig'])
                        mg = log(abs(X[l]/(F[l]*G[l])))
                        hp = array([mg + 20*abs(sqrt(abs(array(fh(H[l])[0])*(sg))))])
                        hp = array([1.0])
                        ip = LSP(fh, hp, df=dfh, show=False,diffInt = 1.5e-8, xtol = 1.5e-8, ftol = 1.5e-8)
                        ip.iprint=-5
                        r = ip.solve('nlp:ralg')
                        hip = r.xf
                        hm = array([mg - 20*abs(sqrt(abs(array(fh(H[l])[0])*(sg))))])
                        hm = array([-1.0])
                        ip = LSP(fh, hm, df=dfh, show=False, diffInt = 1.5e-8, xtol = 1.5e-8, ftol = 1.5e-8)
                        ip.iprint=-5
                        r = ip.solve('nlp:ralg')
                        him = r.xf
                        accept = (abs(hip-him)>0.5) # found a real interval
                        if accept:
                            hinew = rand(1)*abs(hip-him) + min(hip,him)
                        else:
                            xl=H[l]
                            xr=
                                hinew = rand(1) - 0.5 + H[l]
                                if fh(hinew)>0:
                                    accept = True
                        H[l]=hinew

            Y[0:len(X)]=F
            Y[len(X):2*len(X)]=G
            Y[2*len(X):3*len(X)]=H
            S[:,k]= Y
        self.stateCache[:,i]=Y
        return S
                        
                        
                        
                    
                    
                    

                

    def primary2array(self):
        ret = array([])
        if 'f' in self.primary:
            ret = hstack((ret,self.param['f'].primary2array()))
        if 'g' in self.primary:
            ret = hstack((ret,self.param['g'].primary2array()))
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
        

    def jointLogLikAll(self,data,F,G,H):
        MUO = F*G*exp(H)
        n,m = data.size()
        self.param['lik'].param['mu'] = MUO
        LL= self.param['lik'].loglik(data)
        LL = LL + self.param['f'].loglik(Data(kron(F,ones((m,1))).T))
        LL = LL + self.param['g'].loglik(Data(kron(G,ones((m,1))).T))
        LL = LL + self.param['h'].loglik(Data(kron(H,ones((m,1))).T))
        return LL


    def jointLogLik(self,X,F,G,H):
        MUO = F*G*exp(H)
        self.param['lik'].param['mu'] = MUO
        LL= self.param['lik'].loglik(Data(X.reshape((len(F),1))))
        LL = LL + self.param['f'].loglik(Data(F.reshape((len(F),1))))
        LL = LL + self.param['g'].loglik(Data(G.reshape((len(F),1))))
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


    def gradJointLogLik(self,X,F,G,H):
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
        
        
    def loglik(self,data,method=None):
        """
        Approximately calculates the likelihood by integrating the laplace approximation for each single datapoint.
        
        """
        LL=array([self.loglikSingle(data,i,method) for i in range(data.X.shape[1])  ])
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


    def dldtheta(self,data,method=None):
        n,m=data.size()
        pf = self.param['f'].primary2array()
        pg = self.param['g'].primary2array()
        ph = self.param['h'].primary2array()
        gn = len(pf)+len(pg)+len(ph)
        grad = zeros((gn,m))
        for i,x in enumerate(data.X.T):
            grad[:,i] = self.dldthetaSingle(data,i,method=method)
        return grad
        

    def dldthetaSingle(self,data,i,method=None):
        """
        
        Computes the gradient of the (approximate) likelihood for a single datapoint.

        :param X: datapoint
        :type X: numpy.array

        """
        if method==None:
            method = 'laplace'
        if method =='laplace':
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
            grad = zeros((gn))

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
            # do MCMCM stufdf
            grad = zeros(5)
        return grad
          
        
    def loglikSingle(self,data,i,method=None):
        """
        Computes the (approximate) log-likelihood for a single
        datapoint via the laplace approximation.

        :param X: datapoint
        :type X:  numpy.array
        
        """
        if method == None:
            method = 'laplace'
        if method=='laplace':
            # maximal likely hidden functions
            F,G,H = self.getMAP(data,i)


            # negative (inverse) hessian at that point as approximate covariance
            nhes = -self.hessianJoint(data.X[:,i],F,G,H) 

            # for log-det of the hessian/covariance, we add a ridge to avoid numerical instability
            s = svd(nhes + eye(nhes.shape[0])*1e-05,compute_uv=False)

            # and now use the laplace approximation 
            LL = self.jointLogLik(data.X[:,i],F,G,H) + 0.5*log(2*pi)*3*len(F) - 0.5*sum((log(s)))
        else:
            # do MCMC
            LL = zeros(4)
        return LL
        


    def initStateCache(self,data):
        n,m = data.size()
        cacheSz = len(self.primary)*self.param['n']
        self.stateCache = ones((cacheSz,m))

        
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
            method = 'laplace'
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
        theta = optimize.fmin_bfgs(f,theta0,df,callback=callb)

            

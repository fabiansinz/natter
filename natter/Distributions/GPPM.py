from __future__ import division
from Distribution import Distribution
from numpy import zeros,eye,kron,dot,reshape,ones,log,pi,sum,diag,where,tril,hstack,squeeze,array,vstack,outer,exp,sqrt
from numpy.random import randn
from numpy.linalg import cholesky,inv,solve,svd
from natter.DataModule import Data
from scipy import optimize
from Gaussian import Gaussian
from natter.Auxiliary.Errors import InitializationError
from scipy.optimize import check_grad


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
        return Data(F*G*exp(G) + N, str(m) + ' sapmles from a ' + self.name)



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
        m=1
        self.param['lik'].param['mu'] = MUO
        LL= self.param['lik'].loglik(Data(X.reshape((len(F),1))))
        LL = LL + self.param['f'].loglik(Data(kron(F,ones((m,1))).T))
        LL = LL + self.param['g'].loglik(Data(kron(G,ones((m,1))).T))
        LL = LL + self.param['h'].loglik(Data(kron(H,ones((m,1))).T))
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
        hes[0:sz,0:sz] = -CF - diag(G**2*exp(2*H))
        hes[sz:2*sz,sz:2*sz] = -CG - diag(F**2*exp(2*H))
        hes[2*sz:3*sz,2*sz:3*sz] = -CH + diag(X - F*G*exp(H) -2*F**2*G**2*exp(2*H)*H)
        return hes
        
        
    def loglik(self,data):
        """
        Approximately calculates the likelihood by integrating the laplace approximation for each single datapoint.
        
        """
        LL=array([self.loglikSingle(x) for x in data.X.T  ])
        return LL


    def getMAP(self,X):
        FI = self.param['f'].sample(1).X.flatten()
        GI = self.param['g'].sample(1).X.flatten()
        HI = self.param['h'].sample(1).X.flatten()
        Y = hstack((FI,GI,HI))
        def f(Y):
            FU = Y[0:len(FI)]
            GU = Y[len(FI):2*len(FI)]
            HU = Y[2*len(FI):3*len(FI)]
            return -self.jointLogLik(X,FU,GU,HU)
        def df(Y):
            FU = Y[0:len(FI)]
            GU = Y[len(FI):2*len(FI)]
            HU = Y[2*len(FI):3*len(FI)]
            return -self.gradJointLogLik(X,FU,GU,HU)
        def callb(theta):
            err = check_grad(f,df,theta)
            print "Error in gradient: " ,err
        Y = optimize.fmin_bfgs(f,Y,df,disp=0)
        F = Y[0:len(FI)]
        G = Y[len(FI):2*len(FI)]
        H = Y[2*len(FI):3*len(FI)]
        return F,G,H

    def dldtheta(self,data):
        n,m=data.size()
        pf = self.param['f'].primary2array()
        pg = self.param['g'].primary2array()
        ph = self.param['h'].primary2array()
        gn = len(pf)+len(pg)+len(ph)
        grad = zeros((gn,m))
        for i,x in enumerate(data.X.T):
            grad[:,i] = self.dldthetaSingle(x)
        return grad
        
    def dldthetaSingle(self,X):
        """
        computes the gradient of the (approximate) likelihood for a single datapoint.
        """
        gn=0
        if 'f' in self.primary:
            pf = self.param['f'].primary2array()
            nf = self.param['f'].param['n']
            gn = gn+len(pf)
        if 'g' in self.primary:
            pg = self.param['g'].primary2array()
            ng = self.param['g'].param['n']
            gn = gn+len(pf)
        if 'h' in self.primary:
            ph = self.param['h'].primary2array()
            nh = self.param['h'].param['n']
            gn = gn + len(ph)
        grad = zeros((gn))
        F,G,H = self.getMAP(X)
        nhes  = -self.hessianJoint(X,F,G,H)
        iHess = inv(nhes+ eye(nhes.shape[0])*1e-05)
        o =0
        oh = 0
        if 'f' in self.primary:
            grad[o:o+len(pf)]= (2*dot( self.param['f'].cholP.T,iHess[oh:oh+nf,oh:oh+nf]))[self.param['f'].I]
            o = o+len(pf)
            oh = oh+nf
        if 'g' in self.primary:
            grad[o:o+len(pg)]= (2*dot( self.param['g'].cholP.T,iHess[oh:oh+ng,oh:oh+ng]))[self.param['g'].I]
            o=o+len(pg)
            oh+=ng
        if 'h' in self.primary:
            grad[o:o+len(ph)]= (2*dot( self.param['h'].cholP.T,iHess[oh:oh+nh,oh:oh+nh]))[self.param['h'].I]
        return grad
          
        
        
    def loglikSingle(self,X):
        F,G,H = self.getMAP(X)
        nhes = -self.hessianJoint(X,F,G,H)
        s = svd(nhes + eye(nhes.shape[0])*1e-05,compute_uv=False)
        LL = self.jointLogLik(X,F,G,H) + 0.5*log(2*pi)*3*len(F) - sum((log(s)))
        return LL
        
        
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

        def f(theta):
            self.array2primary(theta)
            LL = self.loglik(data)
            return -sum(LL)
        def df(theta):
            self.array2primary(theta)
            grad = -self.dldtheta(data)
            return sum(grad,axis=1)
        def callb(theta):
            err = check_grad(f,df,theta)
            print "Error in gradient: " ,err

        theta0=self.primary2array()
        theta = optimize.fmin_bfgs(f,theta0,df,callback=callb)

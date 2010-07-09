from __future__ import division
from Distribution import Distribution
from numpy import zeros, eye, kron, dot, reshape,ones, log,pi, sum, diag,  where,  tril, hstack, squeeze, array,vstack,outer,sqrt,exp,isnan,isfinite
from numpy.linalg import cholesky, inv, solve
from numpy.random import randn, wald,rand
from natter.DataModule import Data 
#from natter.Auxiliary import debug
from scipy import optimize,stats
from scipy.special import kv,kvp
from natter.Auxiliary import fillDict


# def nanmean(x,axis=0):
#     y = x
#     y[x[isnan(x)]]=0
#     return sum(y,axis=axis)

    



def lognigpdf(x,alpha,beta,delta,mu):
    qx = sqrt(delta**2 + (x-mu)**2)
    return log(delta) + log(alpha) - log(pi) -log(qx) + log(kv(1.,alpha*qx))

def randig(n,delta,gamma):
    V= randn(n)**2
    x1 = delta / gamma * ones(n) + 1 / (2 * gamma**2) * (V + sqrt(4 * gamma * delta * V + V**2))
    x2 = delta / gamma * ones(n) + 1 / (2 * gamma**2) * (V - sqrt(4 * gamma * delta * V + V**2))
    z = rand(n)
    p1 = (delta * ones(n))/ (delta * ones(n) + gamma * x1)
    p2 = ones(n) - p1
    C = (z < p1);
    x1[z>p1] = x2[z>p1]
    return x1

class NIG(Distribution):
    """
    Base class for a  normal inverse Gaussian distribution.
    This distribution is given by the following mean-variance scale mixture:

    :math:`p(x) = \int \mathcal{N}(x|\mu + \beta z \Gamma, z\Gamma) IG(z,\delta^2,\alpha^2 - \beta^\top \Gamma\beta) d z`




    TODO: much todo here...  Gamma should be cholesky
    factor implement EM as well as ML estimation of paarameters """

    def __init__(self, param=None):
        """
        we should represent:
        la = log(alpha)
        ld = log(delta)
        lb = 
        """
        Distribution.__init__(self)
        defaultParam = {'alpha':1.0,
                        'beta':0.0,
                        'mu':0.0,
                        'delta':1.0}
        params = fillDict(defaultParam,param)
        for key in params.keys():
            self.param[key]=params[key]
        self.name = "normal inverse Gaussian"
        self.primary = ['alpha','beta','mu','delta']
                     
        
    def loglik(self,data):
        q = log(self.param['delta']) + log(self.param['alpha']) - log(pi)  +  self.param['delta'] * sqrt(self.param['alpha']**2 - self.param['beta']**2)
        z = 0.5*log(self.param['delta']**2 + (data.X-self.param['mu'])**2)
        K = log(kv(1, (self.param['alpha'] * exp(z))))
        y = q-z + K + self.param['beta'] * (data.X - self.param['mu'])
        return y

        
    def sample(self,m):
        """
        draw m samples from the NIG.
        """
        Z = randig(m,self.param['delta'],sqrt(self.param['alpha']**2 - self.param['beta']**2))
        Y = randn(m)
        return Data(sqrt(Z)*Y + self.param['mu']*ones(m) + self.param['beta']*Z )


    def primary2array(self):
        ret = array([])
        for key in self.primary:
            ret = hstack((ret,self.param[key]))
        return ret

    def array2primary(self,arr):
        for k,key in enumerate(self.primary):
            self.param[key]=arr[k]

    def dldtheta(self,data):
        n,m = data.size()
        a=self.param['alpha']
        b=self.param['beta']
        mu = self.param['mu']
        d  = self.param['delta']
        z = 0.5*log(d**2 + (data.X-mu)**2)
        grad = zeros((len(self.primary),m))
        for k,key in enumerate(self.primary):
            if key == 'alpha':
                grad[k,:]=1/a + d*a/sqrt(a**2 -b**2) + 1/kv(1,a*exp(z))*kvp(1,a*exp(z))*exp(z)
            elif key=='beta':
                grad[k,:]= - d*b/sqrt(a**2-b**2) + (data.X-mu)
            elif key =='delta':
                dz = d/(d**2 + (data.X - mu)**2) 
                grad[k,:]=1/d +sqrt(a**2-b**2)- dz + 1/kv(1,a*exp(z))*kvp(1,a*exp(z))*a*exp(z)*dz
            elif key=='mu':
                dz = -(data.X-mu)/(d**2+(data.X-mu)**2)
                grad[k,:]=-dz +1/kv(1,a*exp(z))*kvp(1,a*exp(z))*a*exp(z)*dz -b
        return grad
    
    def estimate(self,data):
        """
        TODO: initialize parameters according to moments.
        """
        def f(arr):
            self.array2primary(arr)
            return -sum(self.loglik(data))
        def df(arr):
            self.array2primary(arr)
            return -sum(self.dldtheta(data),axis=1)
        arr0=self.primary2array()
        arropt = optimize.fmin_bfgs(f,arr0,df)

    
        
        
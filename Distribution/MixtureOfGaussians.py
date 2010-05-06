import Distribution
import Data
import numpy as np
import scipy as sp
import mpmath
from scipy import stats
import sys
from Auxiliary.Numerics import logsumexp

class MixtureOfGaussians(Distribution.Distribution):
    '''
      Mixture Of Gaussians

      Parameters and their defaults are:
         K:    Number of mixture components (default K=3)
         s:    Standard deviations of the single mixture components
         mu:   Means of the single mixture components
         pi:   Prior of mixture components 
    '''

    
    def __init__(self,param=None):
        self.name = 'Mixture of Gaussians'
        self.param = {'K':3,'s':5.0*np.random.rand(3),'mu':10.0*np.random.randn(3),'pi':np.random.rand(3) }
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
            self.param['K'] = int(self.param['K'])
            if not param.has_key('s'):
                self.param['s'] = 5.0*np.random.rand(self.param['K'])
            if not param.has_key('mu'):
                self.param['mu'] = 10.0*np.random.randn(self.param['K'])
            if not param.has_key('pi'):
                self.param['pi'] = np.random.rand(self.param['K'])
        self.param['pi'] /= np.sum(self.param['pi'])
                
                                            

        
    def sample(self,m):
        '''SAMPLE(M)

           samples M examples from the distribution.
        '''
        u = np.random.rand(m)
        cum_pi = np.cumsum(self.param['pi'])
        k = np.array(m*[0])
        for i in range(m):
            for j in range(len(cum_pi)):
                if u[i] < cum_pi[j]:
                    k[i] = j
                    break
        k = tuple(k)
        return Data.Data(np.random.randn(m)*self.param['s'].take(k) + self.param['mu'].take(k),str(m) + ' sample from ' + self.name)
                

    def pdf(self,dat):
        '''PDF(DAT)
        
           computes the probability density function on the data points in DAT.
        '''
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.pdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret


    def __kloglik(self,dat,k):
        return -.5*np.log(np.pi*2.0) - np.log(self.param['s'][k]) - (dat.X-self.param['mu'][k])**2 / (2.0*self.param['s'][k]**2.0)
        


    def loglik(self,dat):
        '''LOGLIK(DAT)
        
           computes the loglikelihood on the data points in DAT.
        '''
        ret = np.zeros((self.param['K'],dat.size(1)))
        for k in range(self.param['K']):
            ret[k,:]  = np.log(self.param['pi'][k]) + np.squeeze(self.__kloglik(dat,k))
        return logsumexp(ret,0)




    def dldx(self,dat):
        ret = 0.0*dat.X
        tmp = 0.0*dat.X
        for k in range(self.param['K']):
            tmp += self.param['pi'][k] * stats.norm.pdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
            ret +=  self.param['pi'][k] * stats.norm.pdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])\
                   * -self.param['s'][k]**(-2.0)  * (dat.X-self.param['mu'][k])
        ret /= tmp
        return ret


    def estimate(self,dat,which=None, errTol=1e-4,maxiter=1000):
        print "\tEstimating Mixture of Gaussians with EM ..."
        errTol=1e-5


        if which == None:
            which = self.param.keys()


        K=self.param['K']
        mu = self.param['mu'].copy()
        s = self.param['s'].copy()
        m = dat.size(1)
        p = self.param['pi'].copy()
        X = dat.X
        
        H = np.zeros((K,m))
        ALLold = ALL = np.Inf

        nr = np.floor(m/K)
        for k in range(K):
            mu[k] = np.mean(X[k*nr:(k+1)*nr+1])

        for i in range(maxiter):
            ALLold = ALL
            sumH = np.zeros((1,m))
            for j in range(K):
                if p[j] < 1e-3:
                    p[j] = 1e-3
                if s[j] < 1e-3:
                    s[j] = 1e-3
            # E-Step
            # the next few lines have been transferred to the log-domain for numerical stability
            for k in range(K):
                H[k,:] = np.log(p[k]) + np.squeeze(-.5*np.log(np.pi*2.0) - np.log(s[k]) - (dat.X-mu[k])**2 / (2.0*s[k]**2.0)) 

            sumH = logsumexp(H,0)
            for k in range(K):
                H[k,:] = H[k,:] - sumH

            H = np.exp(H) # leave log-domain here


            p = np.squeeze(np.mean(H,1))
            sumHk = np.sum(H,1)

            mu = np.dot(H,X.transpose())/sumHk
            for k in range(K):
                s[k] = np.sqrt(np.sum(H[k,:]*(X-mu[k])**2)/sumHk[k])

            if which.count('mu') > 0:
                self.param['mu'] = mu
            if which.count('pi') > 0:
                self.param['pi'] = p
            if which.count('s') > 0:
                self.param['s'] = s
            if i >= 2:
                ALL = self.all(dat)
                print "\r\t Mixture Of Gaussians ALL: %.8f [Bits]" % ALL,
                sys.stdout.flush()
                if np.abs(ALLold-ALL)<errTol:
                    break
        print "\t[EM finished]"

        
        
    def cdf(self,dat):
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.cdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret
        

        

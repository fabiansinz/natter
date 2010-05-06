import Distribution
import Data
import numpy as np
import mpmath
from scipy import stats
import sys
import MixtureOfGaussians
from Auxiliary.Numerics import logsumexp

class MixtureOfLogNormals(Distribution.Distribution):
    '''
      Mixture Of LogNormals

      Parameters and their defaults are:
         K:    Number of mixture components (default K=3)
         s:    Standard deviations of the single mixture components
         mu:   Means of the single mixture components
         pi:   Prior of mixture components 
    '''

    
    def __init__(self,param=None):
        self.name = 'Mixture of LogNormals'
        
        self.param = {'K':3,'s':3.0*np.array(3*[0.2]),'mu':3.0*np.random.randn(3),'pi':np.random.rand(3) }
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
            self.param['K'] = int(self.param['K'])
            if not param.has_key('s'):
                self.param['s'] = 3.0*np.array(self.param['K']*[0.2])
            if not param.has_key('mu'):
                self.param['mu'] = 3.0*np.random.randn(self.param['K'])
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
        return Data.Data(np.exp(np.random.randn(m)*self.param['s'].take(k) + self.param['mu'].take(k)),str(m) + ' sample from ' + self.name)
                

    def pdf(self,dat):
        '''PDF(DAT)
        
           computes the probability density function on the data points in DAT.
        '''
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.pdf(np.log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k]) / dat.X
        return ret


    def loglik(self,dat):
        '''LOGLIK(DAT)
        
           computes the loglikelihood on the data points in DAT.
        '''
        ret = np.zeros((self.param['K'],dat.size(1)))
        for k in xrange(self.param['K']):
            ret[k,:]  = np.log(self.param['pi'][k]) + np.squeeze(self.__kloglik(dat,k))
        return logsumexp(ret,0)
        
    def __kloglik(self,dat,k):
        return -np.log(dat.X) - .5*np.log(np.pi*2.0) - np.log(self.param['s'][k]) \
            - .5 / self.param['s'][k]**2.0 * (np.log(dat.X) - self.param['mu'][k])**2

    def dldx(self,dat):
        ret = 0.0*dat.X
        tmp = 0.0*dat.X
        for k in range(self.param['K']):
            tmp += self.param['pi'][k] * stats.norm.pdf(np.log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k]) / dat.X
            ret += - self.param['pi'][k] * stats.norm.pdf(np.log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k]) / dat.X \
                   * (1.0/dat.X + (np.log(dat.X) - self.param['mu'][k])/(self.param['s'][k]**2.0 * dat.X))
        ret /= tmp
        return ret
        


    def estimate(self,dat,which=None, errTol=1e-4,maxiter=1000):
        dummy = MixtureOfGaussians.MixtureOfGaussians(self.param)
        #dummy.histogram(Data.Data(np.log(dat.X)))
        dummy.estimate(Data.Data(np.log(dat.X)),which, errTol, maxiter)
        self.param = dummy.param

        
    def cdf(self,dat):
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.cdf(np.log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret
        

        

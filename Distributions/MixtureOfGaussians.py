from Distributions import Distribution
from DataModule import Data
from numpy import sum, cumsum, array, log, pi, zeros, squeeze, Inf, floor, mean, exp, sum, dot, sqrt, abs
from numpy.random import rand, randn 
from scipy import stats
import sys
from Auxiliary.Numerics import logsumexp

class MixtureOfGaussians(Distribution):
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
        self.param = {'K':3,'s':5.0*rand(3),'mu':10.0*randn(3),'pi':rand(3) }
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
            self.param['K'] = int(self.param['K'])
            if not param.has_key('s'):
                self.param['s'] = 5.0*rand(self.param['K'])
            if not param.has_key('mu'):
                self.param['mu'] = 10.0*randn(self.param['K'])
            if not param.has_key('pi'):
                self.param['pi'] = rand(self.param['K'])
        self.param['pi'] /= sum(self.param['pi'])
        self.primary = ['pi','mu','s']
                
                                            

        
    def sample(self,m):
        '''SAMPLE(M)

           samples M examples from the distribution.
        '''
        u = rand(m)
        cum_pi = cumsum(self.param['pi'])
        k = array(m*[0])
        for i in range(m):
            for j in range(len(cum_pi)):
                if u[i] < cum_pi[j]:
                    k[i] = j
                    break
        k = tuple(k)
        return Data(randn(m)*self.param['s'].take(k) + self.param['mu'].take(k),str(m) + ' sample from ' + self.name)
                

    def pdf(self,dat):
        '''PDF(DAT)
        
           computes the probability density function on the data points in DAT.
        '''
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.pdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret


    def __kloglik(self,dat,k):
        return -.5*log(pi*2.0) - log(self.param['s'][k]) - (dat.X-self.param['mu'][k])**2 / (2.0*self.param['s'][k]**2.0)
        


    def loglik(self,dat):
        '''LOGLIK(DAT)
        
           computes the loglikelihood on the data points in DAT.
        '''
        ret = zeros((self.param['K'],dat.size(1)))
        for k in range(self.param['K']):
            ret[k,:]  = log(self.param['pi'][k]) + squeeze(self.__kloglik(dat,k))
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


    def estimate(self,dat, errTol=1e-4,maxiter=1000):
        print "\tEstimating Mixture of Gaussians with EM ..."
        errTol=1e-5




        K=self.param['K']
        mu = self.param['mu'].copy()
        s = self.param['s'].copy()
        m = dat.size(1)
        p = self.param['pi'].copy()
        X = dat.X
        
        H = zeros((K,m))
        ALLold = ALL = Inf

        nr = floor(m/K)
        for k in range(K):
            mu[k] = mean(X[k*nr:(k+1)*nr+1])

        for i in range(maxiter):
            ALLold = ALL
            sumH = zeros((1,m))
            for j in range(K):
                if p[j] < 1e-3:
                    p[j] = 1e-3
                if s[j] < 1e-3:
                    s[j] = 1e-3
            # E-Step
            # the next few lines have been transferred to the log-domain for numerical stability
            for k in range(K):
                H[k,:] = log(p[k]) + squeeze(-.5*log(pi*2.0) - log(s[k]) - (dat.X-mu[k])**2 / (2.0*s[k]**2.0)) 

            sumH = logsumexp(H,0)
            for k in range(K):
                H[k,:] = H[k,:] - sumH

            H = exp(H) # leave log-domain here


            p = squeeze(mean(H,1))
            sumHk = sum(H,1)

            mu = dot(H,X.transpose())/sumHk
            for k in range(K):
                s[k] = sqrt(sum(H[k,:]*(X-mu[k])**2)/sumHk[k])

            if 'mu' in self.primary:
                self.param['mu'] = mu
            if 'pi' in self.primary:
                self.param['pi'] = p
            if 's' in self.primary:
                self.param['s'] = s
            if i >= 2:
                ALL = self.all(dat)
                print "\r\t Mixture Of Gaussians ALL: %.8f [Bits]" % ALL,
                sys.stdout.flush()
                if abs(ALLold-ALL)<errTol:
                    break
        print "\t[EM finished]"

        
        
    def cdf(self,dat):
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.cdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret
        

        

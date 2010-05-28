from Distribution import Distribution
from MixtureOfGaussians import MixtureOfGaussians
from natter.DataModule import Data
from numpy import array, sum,cumsum, exp, log, zeros, squeeze, pi
from numpy.random import randn, rand
from scipy.stats import norm
from natter.Auxiliary.Numerics import logsumexp

class MixtureOfLogNormals(Distribution):
    """
    Mixture of log-Normals

    
    :param param:
        dictionary which might containt parameters for the Mixture of log-Normals
              'K'    :   number of mixtures (default=3)

              's'    :   standard deviations  
   
              'mu'   :   means

              'pi'  :   prior mixture probabilities (default normalized random array of lenth L)

    :type param: dict

    Primary parameters are ['pi','mu','s'].
        
    """
    def __init__(self,param=None):
        self.name = 'Mixture of LogNormals'
        
        self.param = {'K':3,'s':3.0*array(3*[0.2]),'mu':3.0*randn(3),'pi':rand(3) }
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
            self.param['K'] = int(self.param['K'])
            if not param.has_key('s'):
                self.param['s'] = 3.0*array(self.param['K']*[0.2])
            if not param.has_key('mu'):
                self.param['mu'] = 3.0*randn(self.param['K'])
            if not param.has_key('pi'):
                self.param['pi'] = rand(self.param['K'])
        self.param['pi'] /= sum(self.param['pi'])
        self.primary = ['pi','mu','s']

        
    def sample(self,m):
        """

        Samples m samples from the current mixture of log-Normals.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        u = rand(m)
        cum_pi = cumsum(self.param['pi'])
        k = array(m*[0])
        for i in range(m):
            for j in range(len(cum_pi)):
                if u[i] < cum_pi[j]:
                    k[i] = j
                    break
        k = tuple(k)
        return Data(exp(randn(m)*self.param['s'].take(k) + self.param['mu'].take(k)),str(m) + ' sample from ' + self.name)
                

    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points,

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * norm.pdf(log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k]) / dat.X
        return ret


    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        ret = zeros((self.param['K'],dat.size(1)))
        for k in xrange(self.param['K']):
            ret[k,:]  = log(self.param['pi'][k]) + squeeze(self.__kloglik(dat,k))
        return logsumexp(ret,0)
        
    def __kloglik(self,dat,k):
        return -log(dat.X) - .5*log(pi*2.0) - log(self.param['s'][k]) \
            - .5 / self.param['s'][k]**2.0 * (log(dat.X) - self.param['mu'][k])**2

    def dldx(self,dat):
        '''
        Computes the derivative of the log-likelihood w.r.t. the data points in dat.

        :param dat: Data for which the derivatives will be computed.
        :type dat:  natter.DataModule.Data
        :returns:   Array containing the derivatives.
        :rtype:     numpy.array
        '''
        
        ret = 0.0*dat.X
        tmp = 0.0*dat.X
        for k in range(self.param['K']):
            tmp += self.param['pi'][k] * norm.pdf(log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k]) / dat.X
            ret += - self.param['pi'][k] * norm.pdf(log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k]) / dat.X \
                   * (1.0/dat.X + (log(dat.X) - self.param['mu'][k])/(self.param['s'][k]**2.0 * dat.X))
        ret /= tmp
        return ret
        


    def estimate(self,dat, errTol=1e-4,maxiter=1000):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).

        The estimation method uses MixtureOfGaussians.estimate on log(x) to fit the parameters.
        
        :param dat: Data points on which the Mixture of Dirichlet distributions will be estimated.
        :type dat: natter.DataModule.Data
        :param errTol: Stopping criterion for the iteration
        :type method: float
        :param maxiter: maximal number of EM iterations
        :param maxiter: int
        
        '''
        dummy = MixtureOfGaussians(self.param)
        dummy.primary = self.primary
        dummy.estimate(Data(log(dat.X)), errTol, maxiter)
        self.param = dummy.param

        
    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution  function on the data points,

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the quantiles.
        :rtype:    numpy.array
           
        '''
        
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * norm.cdf(log(dat.X),loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret
        

        

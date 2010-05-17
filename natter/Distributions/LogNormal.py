from Distribution import Distribution
from natter.DataModule import Data
from numpy import exp, log, pi, mean, std
from numpy.random import randn
from scipy.stats import norm

class LogNormal(Distribution):
    maxCount = 1000
    Tol = 10.0**-20.0
    
    def __init__(self,param=None):
        '''
        LogNormal distribution constructor.
        
        :param param: Initial parameters for the LogNormal distribution. The LogNormal distribution has parameters *mu* (mean of log x) and *s* (std of log x). The default value for param is {'mu':0.0,'s':1.0}.

        Primary parameters are ['mu','s'].
        
        :type param: dict.
        :returns:  A LogNormal distribution object initialized with the parameters in param.
        
        '''
        self.name = 'Log-Normal Distribution'
        self.param = {'mu':0.0,'s':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['mu','s']


    
        
        
    def sample(self,m):
        """

        Samples m samples from the current GammaP distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """
        return Data(exp(randn(1,m)*self.param['s'] + self.param['mu']) ,str(m) + ' samples from ' + self.name)
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        return -log(dat.X) - .5*log(pi*2.0) - log(self.param['s']) \
               - .5 / self.param['s']**2.0 * (log(dat.X) - self.param['mu'])**2


    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''

        return exp(self.loglik(dat))
        

    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''
        return norm.cdf(log(dat.X),loc=self.param['mu'],scale=self.param['s'])


    def ppf(self,X):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''

        return Data(exp(norm.ppf(X,loc=self.param['mu'],scale=self.param['s'])))


    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the derivatives.
        :rtype:    numpy.array
        
        """        
        return -1.0/dat.X  - 1.0 / self.param['s']**2.0 * (log(dat.X) - self.param['mu']) / dat.X
        

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).

        Estimate fits the log-normal distribution by fitting a Gaussian distribution to log x.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param prange: Range to be search in for the optimal *p*.
        :type prange:  tuple
        
        '''


        if 'mu' in self.primary:
            self.param['mu'] = mean(log(dat.X))
    

        if 's' in self.primary:
            self.param['s'] = std(log(dat.X))


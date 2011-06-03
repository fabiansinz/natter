from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros, sqrt, pi,squeeze, any, isnan,min,array,where
from scipy.stats import truncnorm, norm
from scipy.optimize import fmin_l_bfgs_b
from natter.Auxiliary.Utils import parseParameters
from warnings import warn
from numpy.random import rand
from natter.Auxiliary.Decorators import DataSupportChecker, ArraySupportChecker

class Truncated(Distribution):
    """
    Truncated Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    Implements a wrapper for univariate distributions to be
    truncated. If None is supplied for the parameters, no bound is assumed.

    :param param:
        dictionary which might containt parameters for the Truncated distribution
              'q'     :   Base distribution that is truncated

              'a'     :   Lower bound 

              'b'     :   Upper bound 
              
    :type param: dict

    Primary parameters are ['q'].
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        self.param = {'q':None,'a':None,'b':None}
        param = parseParameters(args,kwargs)
        for k,v in param.items():
            self.param[k] = v
        
        # set default parameters
        self.name = 'Truncated %s Distribution' % (self.param['q'].name,)
        self.primary = []


        
    def sample(self,m):
        """

        Samples m samples from the current Truncated
        distribution. Needs the base distribution to implement ppf.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        dat = self.ppf(rand(m))
        dat.name = "%i samples from a truncated %s" % (m,self.param['q'].name)
        dat.history = ['sampled from a truncated %s' % (self.param['q'].name)]
        return dat
        

    @DataSupportChecker(1,'a','b')
    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        u  = self.param['q'].cdf(Data(array([self.param['a'],self.param['b']])))
        
        return self.param['q'].loglik(dat) - log(u[1] - u[0])
    
    
    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        return exp(self.loglik(dat))

    @DataSupportChecker(1,'a','b')
    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''
        #print dat.X
        u  = self.param['q'].cdf(Data(array([self.param['a'],self.param['b']])))
        return (self.param['q'].cdf(dat)-u[0])/(u[1] - u[0])

    @ArraySupportChecker(1,0.0,1.0)
    def ppf(self,u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        
        p  = self.param['q'].cdf(Data(array([self.param['a'],self.param['b']])))
        u = u*(p[1]-p[0]) + p[0]
        return self.param['q'].ppf(u)

    

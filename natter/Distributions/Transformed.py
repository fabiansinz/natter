from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros, sqrt, pi,squeeze, any, isnan,min,array,where,abs
from scipy.stats import truncnorm, norm
from scipy.optimize import fmin_l_bfgs_b
from natter.Auxiliary.Utils import parseParameters
from warnings import warn
from numpy.random import rand

class Transformed(Distribution):
    """
    Transformed Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    Implements a wrapper for univariate distribution q such that the
    random variable by the transformed distribution is distributed
    according to :math:`y \sim q(f^-1(y)) |dx/dy|`. In
    other words, samples from the transformed distribution are samples
    x from q tranformed with f, i.e. y = f(x).

    Of course, f must be a bijective mapping on the domain of x.

    IMPORTANT: The functions should not modify the Data objects.

    :param param:
        dictionary which might containt parameters for the Transformed distribution
              'q'       :   Base distribution whose random variables are transformed

              'f'       :   forward function (takes DataModule.Data object, returns Data)

              'finv'    :   inverse function (takes DataModule.Data object, returns Data)

              'dfinvdy' :   derivative of the inverse function w.r.t. y (takes DataModule.Data object, returns an (m,) shape array)
              
    :type param: dict

    Primary parameters are ['q'].
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        self.param = {'f':None,'finv':None,'dfinvdy':None,'q':None}
        param = parseParameters(args,kwargs)
        for k,v in param.items():
            self.param[k] = v
        
        # set default parameters
        self.name = 'Transformed %s Distribution' % (self.param['q'].name,)
        self.primary = []


        
    def sample(self,m):
        """

        Samples m samples from the current Transformed
        distribution. Needs the base distribution to implement ppf.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        dat = self.param['f'](self.param['q'].sample(m))
        dat.name = "%i samples from a transformed %s" % (m,self.param['q'].name)
        dat.history = ['sampled from a %s' % (self.param['q'].name,), 'transformed']
        return dat
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        
        return self.param['q'].loglik(self.param['finv'](dat)) + log(abs(squeeze(self.param['dfinvdy'](dat))))
    
    
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
        return self.param['q'].cdf(self.param['finv'](dat))

    def ppf(self,u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        
        return self.param['f'](self.param['q'].ppf(u))

    

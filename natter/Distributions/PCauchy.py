from __future__ import division
from natter.DataModule import Data
from numpy import mean, sum, abs, log,zeros,atleast_2d

from natter.Auxiliary.Utils import parseParameters
from scipy.special import gamma, digamma, gammaln
from Distribution import Distribution
from scipy.optimize import fmin_l_bfgs_b


class PCauchy(Distribution):
    """
      PCauchy Distribution

      The constructor is either called with a dictionary, holding
      the parameters (see below) or directly with the parameter
      assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
      also possible.

      The p-Cauchy distribution in n dimensions is the ratio
      distribution of a Lp-spherically symmetric distribution in n+1
      dimensions.

      :param param:
      dictionary which might containt parameters for the Gamma distribution
              'n'    :    Dimensions of the p-Cauchy distributed random variables. 
      
              'p'    :    Exponent (default p=2.0)
              
              :type param: dict

    Primary parameters are ['p'].
    """

    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        # (special.gamma(1.0/p)/special.gamma(3.0/p))**(p/2.0)
        param = parseParameters(args,kwargs)

        # set default parameters
        self.param = {'n':2, 'p':2.0}
        self.name = 'PCauchy Distribution'
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['p']



    def loglik(self,dat):
        """

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
        """
        n = float(self.param['n'])
        p = float(self.param['p'])
        return n * log(p) + gammaln((n+1)/p) - n*log(2.0) - (n+1)*gammaln(1/p) - (n+1)/p*log(1.0 + dat.norm(p).X.ravel()**p)

    def dldtheta(self,dat):
        """
        Evaluates the gradient of the distribution with respect to the primary parameters.

        :param dat: Data on which the gradient should be evaluated.
        :type dat: DataModule.Data
        :returns:   The gradient
        :rtype:     numpy.array
        
        """

        n = float(self.param['n'])
        p = float(self.param['p'])

        r = atleast_2d(sum(abs(dat.X)**p,axis=0))
        return n/p - (n+1)/p**2.0*(digamma((n+1)/p)-digamma(1/p)) + (n+1)/p**2.0*log(1.0 + r)\
               - (n+1)*sum(abs(dat.X)**p * log(abs(dat.X)),axis=0) / p / (1.0 + r)



    

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly.

        :param dat: Data points on which the NakaRushton distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''

        f = lambda p: self.array2primary(p).all(dat)
        fprime = lambda p: -mean(self.array2primary(p).dldtheta(dat),1) / log(2) / dat.size(0)
        
   
        tmp = fmin_l_bfgs_b(f, self.primary2array(), fprime,  bounds=self.primaryBounds(),factr=.01)[0]
        self.array2primary(tmp)

    def primaryBounds(self):
        return [(1e-6,None)]

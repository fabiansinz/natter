from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros,  sqrt, pi, sign, dot, array, squeeze
from scipy.stats import truncnorm, norm
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import gamma, gammaln, digamma
from natter.Auxiliary.Numerics import owensT
from natter.Auxiliary.Utils import parseParameters
from numpy.linalg import cholesky
from numpy.random import randn

class TModelRadial(Distribution):
    """
    TModelRadial Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the TModelRadial distribution

              'n'     :   dimensionality of the data

              'alpha' :  shape parameter

              'beta'  :    scale parameter
              
    :type param: dict

    Primary parameters are ['alpha','beta'].
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)
        
        # set default parameters
        self.name = 'TModelRadial Distribution'
        self.param = {'beta':1.0,'alpha':1.0,'n':2} 
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['alpha','beta']


    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        alpha = self.param['alpha']
        beta = self.param['beta']
        n = float(self.param['n'])
        return beta*log(alpha) + gammaln(beta+n/2.0) +log(2.0) + (n-1.0)*log(dat.X.ravel()) - gammaln(n/2.0) \
               - (beta+n/2.0)*log(alpha + dat.X.ravel()**2.0) - gammaln(beta)




    def primaryBounds(self):
        return len(self.primary)*[(1e-6,None)]
    
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
        
   
        tmp = fmin_l_bfgs_b(f, self.primary2array(), fprime,  bounds=self.primaryBounds(),factr=10.0)[0]
        self.array2primary(tmp)

    def dldtheta(self,dat):
        """
        Evaluates the gradient of the skewed Gaussian loglikelihood with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        
        """        
        alpha = self.param['alpha']
        beta = self.param['beta']
        n = float(self.param['n'])
        
        grad = zeros((len(self.primary),dat.numex()))
        for i,k in enumerate(self.primary):
            if k == 'alpha':
                grad[i,:] = beta/alpha - (beta+n/2.0)/(alpha + dat.X.ravel()**2.0)
            if k == 'beta':
                grad[i,:] = log(alpha) + digamma(beta+n/2.0) - digamma(beta) - log(alpha + dat.X.ravel()**2.0)
        
        return grad

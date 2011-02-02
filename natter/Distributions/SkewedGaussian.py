from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros,  sqrt, pi, sign, dot, array, squeeze
from scipy.stats import truncnorm, norm
from scipy.optimize import fmin_l_bfgs_b
from natter.Auxiliary.Numerics import owensT
from natter.Auxiliary.Utils import parseParameters
from numpy.linalg import cholesky
from numpy.random import randn

class SkewedGaussian(Distribution):
    """
    SkewedGaussian Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the SkewedGaussian distribution

              'mu'    :   mean (default mu=0)

              'sigma' :   standard deviation (default sigma=1.0)

              'alpha' :   skewness parameters (default alpha=1)
              
    :type param: dict

    Primary parameters are ['mu','sigma','alpha'].
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)
        
        # set default parameters
        self.name = 'SkewedGaussian Distribution'
        self.param = {'mu':1.0,'sigma':1.0,'alpha':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['mu','sigma','alpha']


        
    def sample(self,m):
        """

        Samples m samples from the current SkewedGaussian distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        rho = self.param['alpha']/sqrt(self.param['alpha']**2 + 1)
        u = dot(cholesky(array([[1,rho],[rho,1]])),randn(2,m))
        
        return Data(u[0,:]*sign(u[1,:])*self.param['sigma']+self.param['mu'],'%i samples from %s' % (m,self.name))
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        return log(self.pdf(dat))
    
    
    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        mu = self.param['mu']
        sigma = self.param['sigma']
        alpha = self.param['alpha']
        s = lambda y: (y-mu)/sigma
        phi = norm.pdf
        Phi = norm.cdf
        
        return squeeze((2.0/sigma*phi(s(dat.X))*Phi(alpha*s(dat.X))) )


    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''
        mu = self.param['mu']
        sigma = self.param['sigma']
        alpha = self.param['alpha']
        s = lambda y: (y-mu)/sigma
        Phi = norm.cdf
        return squeeze(Phi(s(dat.X)) - 2*owensT(s(dat.X), alpha).T)
    


    
    def primary2array(self):
        """
        converts primary parameters into an array.
        """
        ret = zeros(len(self.primary))
        for ind,key in enumerate(self.primary):
            ret[ind]=self.param[key]
        return ret

    def array2primary(self,arr):
        """
        Converts the given array into primary parameters.

        :returns: The object itself.
        :rtype: natter.Distributions.SkewedGaussian
            
        """
        ind = 0
        if 'mu' in self.primary:
            self.param['mu'] = arr[ind]
            ind += 1
        if 'sigma' in self.primary:
            self.param['sigma'] = arr[ind]
            ind += 1
        if 'alpha' in self.primary:
            self.param['alpha'] = arr[ind]
            ind += 1
            
        return self
            
    
    

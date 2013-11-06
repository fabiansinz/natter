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
        :type m: int.
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
        mu = self.param['mu']
        sigma = self.param['sigma']
        alpha = self.param['alpha']
        s = lambda y: (y-mu)/sigma
        logphi = norm.logpdf
        logPhi = norm.logcdf
        
        return ( log(2.0) - log(sigma) +logphi(s(dat.X)) + logPhi(alpha*s(dat.X)) ).ravel()
    
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

    def primaryBounds(self):
        """
        Provide bounds on the primary parameters. Returns
        None, if the parameter is unbounded in that direction.

        :returns: bounds on the primary parameters
        :rtype: list of tuples containing the single lower and upper bounds
        """
        ret = []
        for k in self.primary:
            if k == 'mu':
                ret.append((None,None))
            else:
                ret.append((1e-6,None))
        return ret
    
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

        :param dat: Data on which the gradient should be evaluated.
        :type dat: DataModule.Data
        
        """        
        s = lambda y: (y-mu)/sigma
        phi = norm.pdf
        Phi = norm.cdf
        mu = self.param['mu']
        sigma = self.param['sigma']
        alpha = self.param['alpha']
        
        grad = zeros((len(self.primary),dat.numex()))
        for i,k in enumerate(self.primary):
            if k == 'mu':
                grad[i,:] = (dat.X-mu)/sigma**2.0 + phi(alpha*s(dat.X))/Phi(alpha*s(dat.X))*-alpha/sigma
            if k == 'sigma':
                grad[i,:] = -1.0/sigma + (dat.X-mu)**2.0/sigma**3.0 + phi(alpha*s(dat.X))/Phi(alpha*s(dat.X))*-alpha*(dat.X-mu)/sigma**2.0
            if k == 'alpha':
                grad[i,:] = phi(alpha*s(dat.X))/Phi(alpha*s(dat.X))*s(dat.X)
        
        return grad

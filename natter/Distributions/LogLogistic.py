from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros,  sqrt, pi, sign, dot, array, squeeze, Inf,atleast_2d,amax,abs
from scipy.optimize import fmin_l_bfgs_b
from natter.Auxiliary.Utils import parseParameters
from numpy.random import rand

class LogLogistic(Distribution):
    """
    LogLogistic Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the LogLogistic distribution

              'alpha'    :  scale parameter

              'beta'     :  shape parameter

              
    :type param: dict

    Primary parameters are ['alpha','beta'].
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)
        
        # set default parameters
        self.name = 'LogLogistic Distribution'
        self.param = {'alpha':1.0,'beta':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['alpha','beta']
    
    def sample(self,m):
        """

        Samples m samples from the current LogLogistic distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return self.ppf(rand(m,))
        
    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        beta = self.param['beta']
        alpha = self.param['alpha']
        return squeeze(log(beta)-beta*log(alpha)+(beta-1.0)*log(dat.X)-2*log(1.0+(dat.X/alpha)**beta))
    
    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        beta = self.param['beta']
        alpha = self.param['alpha']

        return squeeze(beta/alpha*(dat.X/alpha)**(beta-1.0)/(1.0 + (dat.X/alpha)**beta)**2.0)

    def ppf(self, u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param u: Percentiles for which the ppf will be computed.
        :type u: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        beta = self.param['beta']
        alpha = self.param['alpha']
        tmp = 1.0/beta*(log(u) - log(1.0-u)) + log(alpha)
        return Data(exp(tmp),'%s ppf function outputs' % (self.name,))


    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''
        beta = self.param['beta']
        alpha = self.param['alpha']
        return dat.X**beta/(alpha**beta + dat.X**beta)

    def primaryBounds(self):
        return len(self.primary)*[(1e-6,None)]
    
    def estimate(self,dat,method='iteration',tol=1e-12):
        """

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly.

        :param dat: Data points on which the NakaRushton distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param method: String indicating the estimation method (possible: 'iteration' and 'gradient'; default: 'iteration').
        :type method: string
        :param tol: stopping tolerance for iteration method
        :type tol: float
        """

        if method=='gradient':
            f = lambda p: self.array2primary(p).all(dat)
            fprime = lambda p: -mean(self.array2primary(p).dldtheta(dat),1) / log(2) / dat.size(0)
   
            tmp = fmin_l_bfgs_b(f, self.primary2array(), fprime,  bounds=self.primaryBounds(),factr=10.0)[0]
            self.array2primary(tmp)
        elif method=='iteration':
            print "\tFitting primary parameters " + len(self.primary)*"%s " % tuple(self.primary)
            a = self.primary2array()
            a_old = Inf*a
            maxiter = 1000
            i = 0
            while amax(abs(a-a_old)) > tol and i < maxiter:
                # print amax(abs(a-a_old))
                dll = mean(self.dldtheta(dat),axis=1)
                ddll = mean(self.d2ldtheta2(dat),axis=1)
                a_old = a
                a = a**2.0*ddll/(a*ddll + dll)
                a[a <= 0.0] = 1-6
                self.array2primary(a)
                i += 1


    def dldtheta(self,dat):
        """
        Evaluates the gradient of the log-logistic loglikelihood with
        respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        
        """        
        beta = self.param['beta']
        alpha = self.param['alpha']
        
        grad = zeros((len(self.primary),dat.numex()))
        for i,k in enumerate(self.primary):
            if k == 'alpha':
                grad[i,:] = beta*(dat.X**beta - alpha**beta)/alpha/(dat.X**beta + alpha**beta)
            if k == 'beta':
                grad[i,:] = 1.0/beta + (1-(dat.X/alpha)**beta)/(1+(dat.X/alpha)**beta)*(log(dat.X)-log(alpha))
        return grad

    def d2ldtheta2(self,dat):
        """
        Evaluates the second derivative of each single parameter of
        the log-logistic loglikelihood with respect to the primary
        parameters (it only computes the repeated second derivative,
        i.e. the diagonal terms of the Hessian).

        :param data: Data on which the derivative should be evaluated.
        :type data: DataModule.Data
        
        """        
        beta = self.param['beta']
        alpha = self.param['alpha']
        
        grad = zeros((len(self.primary),dat.numex()))
        for i,k in enumerate(self.primary):
            if k == 'alpha':
                grad[i,:] = (-beta**2.0*alpha**beta*(dat.X**beta + alpha**beta) \
                             - beta*(dat.X**beta - alpha**beta)*(dat.X**beta+(beta+1)*alpha**beta))\
                             /alpha**2.0/(dat.X**beta+alpha**beta)**2.0
            if k == 'beta':
                grad[i,:] = -1.0/beta**2.0 - 2.0*(dat.X/alpha)**beta/(1+(dat.X/alpha)**beta)**2.0*(log(dat.X)-log(alpha))**2.0
        return grad

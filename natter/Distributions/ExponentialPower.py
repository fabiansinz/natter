from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, abs, sign, exp, mean, squeeze, any, where, isinf, isnan
from numpy.random import gamma, randn
from scipy.special import gammaln,gammainc, gammaincinv
from scipy.optimize import fminbound
import types


class ExponentialPower(Distribution):
    """
    Exponential Power Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the Exponential Power distribution
              'p'    :    Exponent (default = 1.0)
              
              's':    Scale parameter (default = 1.0)
              
    :type param: dict

    Primary parameters are ['p','s'].
        
    """

    
    maxCount = 10000
    Tol = 10.0**-20.0
    
    def __init__(self, *args,**kwargs):
                # parse parameters correctly
        param = None
        if len(args) > 0:
            param = args[0]
        if kwargs.has_key('param'):
            if param == None:
                param = kwargs['param']
            else:
                for k,v in kwargs['param'].items():
                    param[k] = v
        if len(kwargs)>0:
            if param == None:
                param = kwargs
            else:
                for k,v in kwargs.items():
                    if k != 'param':
                        param[k] = v
        
        # set default parameters

        self.name = 'Exponential Power Distribution'
        self.param = {'p':1.0,'s':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['p','s']

        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        return squeeze(log(self.param['p']) - log(2.0) - 1.0/self.param['p']*log(self.param['s']) \
               -gammaln(1.0/self.param['p']) - abs(dat.X)**self.param['p']/self.param['s'])


    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the ExponentialPower distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array
        
        """
        return squeeze(- sign(dat.X) * abs(dat.X)**(self.param['p']-1) *self.param['p'] / self.param['s'])




    def sample(self,m):
        """

        Samples m samples from the current ExponentialPower distribution.

        :param m: Number of samples to draw.
        :type m: int.
        :returns:  A Data object containing the samples

        """
        z = gamma(1.0/self.param['p'],self.param['s'],(1,m))**(1.0/self.param['p'])
        return Data(sign(randn(1,m))*z, str(m) + ' samples from an exponential power distribution.')
        
    

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
        return squeeze(.5 + 0.5*sign(dat.X)*gammainc(1/self.param['p'],abs(dat.X)**self.param['p'] / self.param['s']))

    def ppf(self,u):
        '''

        Evaluates the percent point function (i.e. the inverse c.d.f.)
        of the current distribution.

        :param u:  Points at which the p.p.f. will be computed.
        :type u: numpy.array
        :returns:  Data object with the resulting points in the domain of this distribution. 
        :rtype:    natter.DataModule.Data
           
        '''
        q = 1/self.param['p']
        s = self.param['s']

        return Data(sign(u-.5) * s**q *gammaincinv(q,abs(2*u-1))**q,'Function values of the p.p.f of %s' % (self.name,))

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).


        :param dat: Data points on which the ExponentialPower distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''

#        ind = ~(any(isinf(dat.X),axis=0) | any(isnan(dat.X),axis=0))
          
        if 'p' in self.primary:
            func = lambda t: self.__objective(t,dat,'s' in self.primary)
            p = fminbound(func, 0.0, 100.0, xtol=1e-7   )
            if type(p) == types.TupleType:
                p = p[0]
            self.param['p'] = p

        if 's' in self.primary:
            self.param['s'] = self.param['p'] * mean(abs(dat.X)**self.param['p'])
        

    def __objective(self,p,dat,est_s):
        self.param['p'] = p
        if est_s:
            self.param['s'] = p * mean(abs(dat.X)**p)
            
        return self.all(dat)

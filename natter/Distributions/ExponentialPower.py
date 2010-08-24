from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, abs, sign, exp, mean
from numpy.random import gamma, randn
from scipy.special import gammaln
from scipy.optimize import fminbound
import types
from copy import deepcopy
class ExponentialPower(Distribution):
    """
    Exponential Power Distribution

    :param param:
        dictionary which might containt parameters for the Exponential Power distribution
              'p'    :    Exponent (default = 1.0)
              
              's':    Scale parameter (default = 1.0)
              
    :type param: dict

    Primary parameters are ['p','s'].
        
    """

    
    maxCount = 10000
    Tol = 10.0**-20.0
    
    def __init__(self,param=None):
        self.name = 'Exponential Power Distribution'
        self.param = {'p':1.0,'s':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['p','s']

        
    def parameters(self,keyval=None):
        """

        Returns the parameters of the distribution as dictionary. This
        dictionary can be used to initialize a new distribution of the
        same type. If *keyval* is set, only the keys or the values of
        this dictionary can be returned (see below). The keys can be
        used to find out which parameters can be accessed via the
        __getitem__ and __setitem__ methods.

        :param keyval: Indicates whether only the keys or the values of the parameter dictionary shall be returned. If keyval=='keys', then only the keys are returned, if keyval=='values' only the values are returned.
        :type keyval: string
        :returns:  A dictionary containing the parameters of the distribution. If keyval is set, a list is returned. 
        :rtype: dict or list
           
        """
        if keyval == None:
            return deepcopy(self.param)
        elif keyval== 'keys':
            return self.param.keys()
        elif keyval == 'values':
            return self.param.value()      

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        return log(self.param['p']) - log(2.0) - 1.0/self.param['p']*log(self.param['s']) \
               -gammaln(1.0/self.param['p']) - abs(dat.X)**self.param['p']/self.param['s']


    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the ExponentialPower distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array
        
        """
        return - sign(dat.X) * abs(dat.X)**(self.param['p']-1) *self.param['p'] / self.param['s']




    def sample(self,m):
        """

        Samples m samples from the current ExponentialPower distribution.

        :param m: Number of samples to draw.
        :type name: int.
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
        


    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).


        :param dat: Data points on which the ExponentialPower distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''


        if 'p' in self.primary:
            func = lambda t: self.__objective(t,dat,'s' in self.primary)
            p = fminbound(func, 0.0, 100.0)
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

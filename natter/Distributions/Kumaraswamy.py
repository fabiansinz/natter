from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros, squeeze
from scipy.stats import beta
from scipy.optimize import fmin_l_bfgs_b
from copy import deepcopy
from natter.Auxiliary.Utils import parseParameters

class Kumaraswamy(Distribution):
    """
    Kumaraswamy Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the Kumaraswamy distribution
              'a'    :    Shape parameter  1 (default = 1.0)
              
              'b'    :    Scale parameter  2 (default = 1.0)

              'B'    :    Upper bound on the range. The distribution is defined on the interval [0,B]. Default is B=1.0.
              
    :type param: dict

    Primary parameters are ['a','b'].
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)
        
        # set default parameters
        self.name = 'Kumaraswamy Distribution'
        self.param = {'a':1.0,'b':1.0,'B':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['a','b']



        
    def sample(self,m):
        """

        Samples m samples from the current Kumaraswamy distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return Data(self.param['B']*beta.rvs(1,self.param['b'],size=m)**(1/self.param['a']),'%i samples from %s' % (m,self.name))
        
    def primaryBounds(self):
        """
        Provide bounds on the primary parameters. Returns
        None, if the parameter is unbounded in that direction.

        :returns: bounds on the primary parameters
        :rtype: list of tuples containing the single lower and upper bounds
        """
        
        return len(self.primary)*[(1e-6,None)]

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        a = self.param['a']
        b = self.param['b']
        B = self.param['B']
        if a == 1:
            return squeeze(log(a*b) - a*b*log(B) + (b-1)*log(B**a - dat.X**a))
        else:
            return squeeze(log(a*b) - a*b*log(B) + (a-1)*log(dat.X) + (b-1)*log(B**a - dat.X**a))

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
        a = self.param['a']
        b = self.param['b']
        B = self.param['B']
        return squeeze(1-(B**a-dat.X**a)**b/B**(a*b))


    def ppf(self,u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        a = self.param['a']
        b = self.param['b']
        B = self.param['B']
        return Data((B**a - B**a*(1-u)**(1/b))**(1/a),'Function values of the ppf of the Kumaraswamy distribution')



    def dldtheta(self,dat):
        """
        Evaluates the gradient of the Kumaraswamy loglikelihood with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        
        """

        m = dat.size(1)
        grad = zeros((len(self.primary),m))
        ind =0
        a = self.param['a']
        b = self.param['b']
        B = self.param['B']
        
        if 'a' in self.primary:
            grad[ind,:] = 1.0/a - b*log(B) + log(dat.X) + (B**a*log(B)-dat.X**a*log(dat.X))*(b-1)/(B**a-dat.X**a)
            ind +=1
        if 'b' in self.primary:
            grad[ind,:] = 1.0/b - a*log(B) + log(B**a - dat.X**a)
        return grad
     



    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly.

        :param dat: Data points on which the Kumaraswamy distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''

        f = lambda p: self.array2primary(p).all(dat)
        fprime = lambda p: -mean(self.array2primary(p).dldtheta(dat),1) / log(2) / dat.size(0)
        
   
#        tmp = fmin_l_bfgs_b(f, self.primary2array(), fprime, bounds=len(self.primary)*[(1e-6,None)],factr=10.0)[0]
        tmp = fmin_l_bfgs_b(f, self.primary2array(), fprime,  bounds=len(self.primary)*[(1e-6,None)],factr=10.0)[0]
        self.array2primary(tmp)
    
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
        :rtype: natter.Distributions.Kumaraswamy
            
        """
        ind = 0
        if 'a' in self.primary:
            self.param['a'] = arr[ind]
            ind += 1
        if 'b' in self.primary:
            self.param['b'] = arr[ind]
            ind += 1
            
        return self
            
    
    

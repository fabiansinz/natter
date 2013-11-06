from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, abs, sign, exp, mean, array, squeeze, zeros
from numpy.random import rand
from scipy.special import gammainc, gammaincinv,  digamma, gammaln
from scipy.special import gamma as gammafunc
from natter.Auxiliary.Utils import parseParameters
from natter.Auxiliary.Numerics import totalDerivativeOfIncGamma
from scipy.optimize import fmin_l_bfgs_b

class TruncatedExponentialPower(Distribution):
    """
    Truncated Exponential Power Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the Truncated Exponential Power distribution
              'p'    :    Exponent (default = 1.0)
              
              's'    :    Scale parameter (default  s=1.0)

              'a'    :    lower support boundary (default a=0)

              'b'    :    upper support boundary (default b=1)
              
    :type param: dict

    Primary parameters are ['p','s'].
        
    """

    
    maxCount = 10000
    Tol = 10.0**-20.0
    
    def __init__(self, *args,**kwargs):
                # parse parameters correctly
        param = parseParameters(args,kwargs)
        
        # set default parameters
        self.name = 'Truncated Exponential Power Distribution'
        self.param = {'p':1.0,'s':1.0,'a':0.0,'b':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['p','s']

        


    def sample(self,m):
        """

        Samples m samples from the current TruncatedExponentialPower distribution.

        :param m: Number of samples to draw.
        :type m: int.
        :returns:  A Data object containing the samples

        """

        u = array([self.param['a'],self.param['b']])
        u = .5 + 0.5*sign(u)*gammainc(1/self.param['p'],abs(u)**self.param['p'] / self.param['s'])
        u = rand(m)
        return Data(self.ppf(u).X, str(m) + ' samples from an exponential power distribution.')
        
    

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
        u = array([self.param['a'],self.param['b']])
        u = .5 + 0.5*sign(u)*gammainc(1/self.param['p'],abs(u)**self.param['p'] / self.param['s'])
        return squeeze(.5 + 0.5*sign(dat.X)*gammainc(1/self.param['p'],abs(dat.X)**self.param['p'] / self.param['s']) - u[0])/(u[1]-u[0])

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
        v = array([self.param['a'],self.param['b']])
        v = .5 + 0.5*sign(v)*gammainc(1/self.param['p'],abs(v)**self.param['p'] / self.param['s'])
        dv = v[1]-v[0]
        return Data(sign(dv*u+v[0]-.5) * s**q *gammaincinv(q,abs(2*(dv*u+v[0])-1))**q,'Percentiles of %s' % (self.name,))

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).


        :param dat: Data points on which the TruncatedExponentialPower distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''
        f = lambda p: self.array2primary(p).all(dat)
        fprime = lambda p: -mean(self.array2primary(p).dldtheta(dat),1) / log(2) / dat.size(0)
        tmp = fmin_l_bfgs_b(f, self.primary2array(), fprime,  bounds=len(self.primary)*[(1e-6,None)],factr=10.0)[0]
        self.array2primary(tmp)        

        
    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        u = array([self.param['a'],self.param['b']])
        u = .5 + 0.5*sign(u)*gammainc(1/self.param['p'],abs(u)**self.param['p'] / self.param['s'])
        return squeeze(log(self.param['p']) - log(2.0) - 1.0/self.param['p']*log(self.param['s']) \
               -gammaln(1.0/self.param['p']) - abs(dat.X)**self.param['p']/self.param['s']) - log(u[1] - u[0])


    def dldtheta(self,dat):
        """
        Evaluates the gradient of the Gamma function with respect to the primary parameters.

        :param dat: Data on which the gradient should be evaluated.
        :type dat: DataModule.Data
        :returns:   The gradient
        :rtype:     numpy.array
        """
        
        m = dat.numex()
        grad = zeros((len(self.primary),m))
        s = self.param['s']
        p = self.param['p']
        u = array([self.param['a'],self.param['b']])
        cab = .5 + 0.5*sign(u)*gammainc(1/p,abs(u)**p /s)
        for ind,key in enumerate(self.primary):
            if key == 's':
                U = -.5*sign(u)*(abs(u) *exp(-abs(u)**p/s))/(s**(1/p+1)*gammafunc(1/p))
                grad[ind,:] = -1.0/p/s + abs(squeeze(dat.X))**p/s**2 - (U[1]-U[0])/(cab[1]-cab[0])
            if key == 'p':
                f = lambda x: 1/x
                df = lambda x: -1/x**2
                g = lambda x,k: abs(u[k])**x/s
                dg = lambda x,k: abs(u[k])**x*log(abs(u[k]))/s
                dIncGamma = array([totalDerivativeOfIncGamma(p,f,lambda v: g(v,0),df,lambda v: dg(v,0)),\
                                   totalDerivativeOfIncGamma(p,f,lambda v: g(v,1),df,lambda v: dg(v,1))])

                U = .5*sign(u) * (dIncGamma/float(gammafunc(1/p)) + gammainc(1/p,abs(u)**p/s)*digamma(1/p)/p**2)
                grad[ind,:] = 1/p + 1/p**2*log(s) + digamma(1/p)*1/p**2 - abs(squeeze(dat.X))**p*log(abs(squeeze(dat.X)))/s - (U[1]-U[0])/(cab[1]-cab[0])

                
        return grad
                
    
    def primary2array(self):
        """
        Converts primary parameters into an array.

        :returns: The parameters in an array
        :rtype:   numpy.array
        """
        ret = zeros(len(self.primary))
        for ind,key in enumerate(self.primary):
            ret[ind]=self.param[key]
        return ret

    def array2primary(self,arr):
        """
        Converts the given array into primary parameters.

        :param arr: Parameters in an array which as the same format as returned by primary2array
        :type arr:  numpy.array
        :returns:   The distribution object
        :rtype:     natter.Distributions.TruncatedExponentialPower
            
        """
        for ind,key in enumerate(self.primary):
            self.param[key]=arr[ind]
        return self

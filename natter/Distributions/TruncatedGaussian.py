from __future__ import division
from Distribution import Distribution
from natter.DataModule import Data
from numpy import log, exp, mean, zeros, sqrt, pi,squeeze, any, isnan
from scipy.stats import truncnorm, norm
from scipy.optimize import fmin_l_bfgs_b
from natter.Auxiliary.Utils import parseParameters
from warnings import warn
from natter.Auxiliary.Decorators import DataSupportChecker

class TruncatedGaussian(Distribution):
    """
    TruncatedGaussian Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the TruncatedGaussian distribution
              'a'     :   Lower boundary (default a=0)
              
              'b'     :   Upper boundary (default b=1)

              'mu'    :   mean (default mu=0)

              'sigma' :   standard deviation (default sigma=1.0)  
              
    :type param: dict

    Primary parameters are ['mu','sigma'].
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)
        
        # set default parameters
        self.name = 'TruncatedGaussian Distribution'
        self.param = {'a':0.0,'b':1.0,'mu':1.0,'sigma':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['mu','sigma']


        
    def sample(self,m):
        """

        Samples m samples from the current TruncatedGaussian distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        a,b = (self.param['a']-self.param['mu'])/self.param['sigma'],(self.param['b']-self.param['mu'])/self.param['sigma']
        return Data(truncnorm.rvs(a,b,loc=self.param['mu'],scale=self.param['sigma'],size=m),'%i samples from %s' % (m,self.name))
        

    @DataSupportChecker(1,'a','b')
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
        mu = self.param['mu']
        sigma = self.param['sigma']
        s = lambda y: (y-mu)/sigma
        phi = norm.pdf
        Phi = norm.cdf        
        ll = squeeze(-log(sigma) + log(phi(s(dat.X))) - log(Phi(s(b))-Phi(s(a))))
        if any(isnan(ll)):
            print self
            print "s(b)" + str(s(b))
            print "s(a)" + str(s(a))
            print "PHI(a)/PHI(b)" + str(Phi(s(a))/Phi(s(b)))
            print "log(PHI(b) - PHI(a))" + str(- log(Phi(s(b))-Phi(s(a))))
            print "log(PHI(s(X)))" + str(log(phi(s(dat.X))))
            print "sigma" + str(sigma)
            print "mu" + str(mu)
            print "a" + str(a)
            print "b" + str(b)
            print "LL" + str(ll)
            raw_input()
        return ll
    
    
    @DataSupportChecker(1,'a','b')
    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        return exp(self.loglik(dat))

    @DataSupportChecker(1,'a','b')
    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''
        a,b = (self.param['a']-self.param['mu'])/self.param['sigma'],(self.param['b']-self.param['mu'])/self.param['sigma']
        return  squeeze(truncnorm.cdf(dat.X,a,b,loc=self.param['mu'],scale=self.param['sigma']))


    def ppf(self,u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        a,b = (self.param['a']-self.param['mu'])/self.param['sigma'],(self.param['b']-self.param['mu'])/self.param['sigma']
        return Data(truncnorm.ppf(u,a,b,loc=self.param['mu'],scale=self.param['sigma']), 'Percentiles from a %s' % (self.name,))


     


    @DataSupportChecker(1,'a','b')
    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly.

        :param dat: Data points on which the TruncatedGaussian distribution will be estimated.
        :type dat: natter.DataModule.Data
        '''
        f = lambda p: self.array2primary(p).all(dat)
        fprime = lambda p: -mean(self.array2primary(p).dldtheta(dat),1) / log(2) / dat.size(0)
        bounds = []
        if 'mu' in self.primary:
            bounds.append((None,None))
        if 'sigma' in self.primary:
            bounds.append((1e-6,None))
        tmp = fmin_l_bfgs_b(f, self.primary2array(), fprime,  bounds=bounds,factr=10.0)[0]
        self.array2primary(tmp)        


    @DataSupportChecker(1,'a','b')
    def dldtheta(self,dat):
        """
        Evaluates the gradient of the TruncatedGaussian loglikelihood with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        
        """

        

        m = dat.size(1)
        grad = zeros((len(self.primary),m))
        ind =0
        a = self.param['a']
        b = self.param['b']
        mu = self.param['mu']
        sigma = self.param['sigma']

        phiprime = lambda x: 1.0/sqrt(2*pi)*exp(-0.5*x**2) * -x
        s = lambda y: (y-mu)/sigma
#        phi = norm.pdf
        phi = lambda z: 1.0/sqrt(2*pi) * exp(-.5* z**2)
        Phi = norm.cdf
        
        if 'mu' in self.primary:
            grad[ind,:] = -1.0/phi(s(dat.X))*phiprime(s(dat.X))/sigma + (phi(s(b))/sigma - phi(s(a))/sigma)/(Phi(s(b)) - Phi(s(a)))
            ind +=1
        if 'sigma' in self.primary:
            grad[ind,:] = -1.0/sigma - 1.0/phi(s(dat.X))*phiprime(s(dat.X)) * s(dat.X)/sigma + (phi(s(b))*s(b)/sigma - phi(s(a))*s(a)/sigma)/(Phi(s(b)) - Phi(s(a)))
        return grad

    
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
        :rtype: natter.Distributions.TruncatedGaussian
            
        """
        ind = 0
        if 'mu' in self.primary:
            self['mu'] = arr[ind]
            ind += 1
        if 'sigma' in self.primary:
            self['sigma'] = arr[ind]
            ind += 1
            
        return self
            
    
    
    def __setitem__(self,key,value):
        if key == 'mu':
            if norm.cdf( (self.param['b']-value)/self.param['sigma']) - norm.cdf( (self.param['a']-value)/self.param['sigma']) < 1e-12:
                warn("TruncatedGaussian.__setitem__: new value of mu too far from the interval [a,b]! leaving it as it is")
            else:
                self.param['mu'] = value
        elif key == 'sigma':
            if norm.cdf( (self.param['b']-self.param['mu'])/value) - norm.cdf( (self.param['a']-self.param['mu'])/value) < 1e-12:
                warn("TruncatedGaussian.__setitem__: new value of sigma too extreme from the interval [a,b]! leaving it as it is")
            else:
                if value <0:
                    warn("TruncatedGaussian.__setitem__: sigma cannot be negative! Setting it to abs(sigma)")
                self.param[key] = abs(value)

        elif key in self.parameters('keys'):
            self.param[key] = value
        else:
            raise KeyError("Parameter %s not defined for %s" % (key,self.name))

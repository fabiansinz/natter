from __future__ import division
from Distribution import Distribution
from natter.Distributions import GammaP, ChiP
from Truncated import Truncated
from natter.DataModule import Data
from numpy import sqrt, array,where,hstack,log,squeeze,reshape,zeros,mean,sum,atleast_2d
from natter.Auxiliary.Utils import parseParameters
from natter.Auxiliary.Numerics import invertMonotonicIncreasingFunction
from scipy.special import gammainc,gammaln
from scipy.special import gamma as gammafunc
from scipy.optimize import fmin_l_bfgs_b


class NakaRushton(Distribution):
    """
    NakaRushton Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    The Naka-Rushton distribution is the radial distribution of a
    Lp-spherically symmetric distribution that gets transformed into a
    (radially truncated if kappa=2) LpGeneralizedNormal with scale s by the
    Naka-Rushton functio r --> (kappa r**(gamma/2 + delta))/sqrt(sigma**2 + r**gamma)

    :param param:
        dictionary which might containt parameters for the NakaRushton distribution
              'sigma'    :    Shape parameter  1 (default = 1.0)
              
              'kappa'    :    Scale parameter (truncation parameter if gamma=2.0)

              'gamma'    :    Shape parameter (must be in (0,2])

              'delta'    :    Shape parameter (must be > 0)

              's'        :    Scale of the resulting LpGeneralizedNormal
 
              'n'        :    Dimensionality of the LpGeneralizedNormal
              
    :type param: dict

    Primary parameters are ['sigma'] if delta=0.0 or ['sigma','kappa','gamma','delta'] otherwise.
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)
        # set default parameters
        self.name = 'NakaRushton Distribution'
        self.param = {'sigma':1.0,'kappa':2.0,'s':1.0,'n':2.0,'p':2.0,'gamma':2.0,'delta':0.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        if param is None or 's' not in param.keys():
            self.param['s'] = .5*(gammafunc(1.0/self.param['p'])/gammafunc(3.0/self.param['p']))**(self.param['p']/2.0)
        if self.param['delta'] == 0.0:
            self.primary = ['sigma'] # gamma would also be possible here, but stays out for backwards compatibility reasons
        else:
            self.primary = ['sigma','kappa','gamma','delta']
            

       
    def sample(self,m):
        """

        Samples m samples from the current NakaRushton distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        sigma,kappa,delta,gamma = self.param['sigma'],self.param['kappa'],self.param['delta'],self.param['gamma']
        if delta == 0.0:
            q =  ChiP(p=self.param['p'],n=self.param['n'],s=self.param['s']*2.0)
            sampleDistribution = Truncated(q =q,a=0.0,b=kappa)
            ret = sampleDistribution.sample(m)
            ret.X = (ret.X*sigma)**(2.0/gamma) /(kappa**2.0 - ret.X**2.0)**(1.0/gamma)
        else:
            sampleDistribution = ChiP(p=self.param['p'],n=self.param['n'],s=self.param['s']*2.0)
            ret = sampleDistribution.sample(m)
            X = ret.X.flatten()
            f = lambda x: x**(gamma/2.0+delta)*kappa/sqrt(sigma**2.0 + x**gamma)
            tmp = invertMonotonicIncreasingFunction(f,X,0.0*X,10.0*X)
            ret.X = atleast_2d(tmp)
            
        ret.history = ['Sampled %i samples from NakaRushton distribution' % (m,)]
        return ret
        

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        sigma,kappa,delta,gamma,n,p,s = \
                        self.param['sigma'],self.param['kappa'],self.param['delta'],\
                        self.param['gamma'],self.param['n'],self.param['p'],self.param['s']
        r = squeeze(dat.X)

        if delta == 0.0:
            addTerm =  - log(gammainc(n/p,kappa**p/2.0/s))
        else:
            addTerm = 0
            
        ll = log(p) + n*log(kappa) + (n*gamma+2.0*n*delta-2.0)/2.0 * log(r)\
             + log(2.0*delta*(r**gamma + sigma**2.0) + gamma*sigma**2.0)\
             -gammaln(n/p) - n/p*log(s) - (n+p)/p * log(2.0) - (n+2.0)/2.0 *log(sigma**2 + r**gamma) \
             - r**(p*gamma/2.0 + p*delta)*kappa**p/2.0/s/(sigma**2+r**gamma)**(p/2.0) + addTerm
 
        return ll

    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''

        sigma,kappa,delta,gamma = self.param['sigma'],self.param['kappa'],self.param['delta'],self.param['gamma']
        
        datf = dat.copy()
        datf.X = kappa * datf.X**(gamma/2+delta) / sqrt(sigma**2.0 + datf.X**gamma)
        
        if delta == 0.0:
            tmp = Truncated(a=0,b=kappa,\
                            q=GammaP(u=(self.param['n']/self.param['p']),s=2*self.param['s'],p=self.param['p']))
        else:
            tmp = GammaP(u=(self.param['n']/self.param['p']),s=2*self.param['s'],p=self.param['p'])
            
        return tmp.cdf(datf)
            
        
    def dldtheta(self,dat):
        """
        Evaluates the gradient of the NakaRushton loglikelihood with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        
        """
        

        grad = zeros((len(self.primary),dat.numex()))
        r = dat.X.ravel()

        sigma,kappa,delta,gamma,n,p,s = \
                        self.param['sigma'],self.param['kappa'],self.param['delta'],\
                        self.param['gamma'],float(self.param['n']),self.param['p'],self.param['s']

        
        for ind,param in enumerate(self.primary):
            if param == 'kappa':
                grad[ind,:] = n/kappa - p*kappa**(p-1.0)*r**(p*gamma/2.0 + p*delta)/2.0/s/(sigma**2.0 + r**gamma)**(p/2.0)
            if param == 'sigma':
                grad[ind,:] = (4.0*delta*sigma+2.0*gamma*sigma)/(2.0*delta*(r**gamma + sigma**2.0) + gamma*sigma**2.0)\
                              - (n+2.0)*sigma / (r**gamma + sigma**2.0)  \
                              + kappa**p * p * sigma*r**(p*gamma/2.0 + p*delta)/ (2.0*s*(sigma**2.0 + r**gamma)**(p/2.0 + 1.0))
            if param == 'gamma':
                grad[ind,:] =  n/2.0*log(r) \
                              + (2.0*delta*r**gamma*log(r) + sigma**2.0)/(2.0*delta*(r**gamma + sigma**2.0) + gamma*sigma**2.0) \
                              - (n+2.0)*r**gamma*log(r)/(2.0*(r**gamma + sigma**2.0)) \
                              - r**(p*gamma/2.0 + p*delta)*kappa**p*p*log(r)/4.0/s \
                              * (1. - r**gamma*(sigma**2.0 + r**gamma)**-1.) / (sigma**2.0 + r**gamma)**(p/2.0)
                
            if param == 'delta':
                grad[ind,:] = n*log(r) + 2.*(r**gamma + sigma**2.0)/(2.*delta*(r**gamma + sigma**2.0) + gamma*sigma**2.) \
                              - kappa**p * p * r**(p*gamma/2. + p*delta) * log(r) / (2. * s * (sigma**2.0 + r**gamma)**(p/2.))
                
            if param == 's':
                grad[ind,:] = -n/p/s + kappa**p * r**(p*gamma/2.0 + p*delta)/2.0/s**2.0/(sigma**2.0 + r**gamma)**(p/2.0)
   
        return grad
     



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
    
    def primary2array(self):
        """
        converts primary parameters into an array.
        """
        ret = zeros(len(self.primary))
        for ind,key in enumerate(self.primary):
            ret[ind]=self.param[key]
        return ret
    
    def primaryBounds(self):
        ret = []
        for ind,key in enumerate(self.primary):
            ret.append((1e-6,None))
        return ret

    def array2primary(self,arr):
        """
        Converts the given array into primary parameters.

        :returns: The object itself.
        :rtype: natter.Distributions.NakaRushton
            
        """
        ind = 0

        for ind,key in enumerate(self.primary):
            self.param[key] = arr[ind]
        
        return self
            

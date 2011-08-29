from __future__ import division
from Distribution import Distribution
from natter.Distributions import GammaP
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
    Naka-Rushton functio r --> (kappa r)/sqrt(sigma**2 + r**gamma)

    :param param:
        dictionary which might containt parameters for the NakaRushton distribution
              'sigma'    :    Shape parameter  1 (default = 1.0)
              
              'kappa'    :    Scale parameter (truncation parameter if gamma=2.0)

              'gamma'    :    Shape parameter (must be in (0,2])

              's'        :    Scale of the resulting LpGeneralizedNormal
 
              'n'        :    Dimensionality of the LpGeneralizedNormal
              
    :type param: dict

    Primary parameters are ['sigma'] if gamma=2.0 or ['sigma','kappa','gamma'] otherwise.
        
    """

    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)
        # set default parameters
        self.name = 'NakaRushton Distribution'
        self.param = {'sigma':1.0,'kappa':2.0,'s':1.0,'n':2.0,'p':2.0,'gamma':2.0}
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        if param is None or 'sigma' not in param.keys():
            self.param['sigma'] = .5*(gammafunc(1.0/self.param['p'])/gammafunc(3.0/self.param['p']))**(self.param['p']/2.0)
        if self.param['gamma'] == 2.0:
            self.primary = ['sigma']
        else:
            self.primary = ['sigma','kappa','gamma']
            

       
    def sample(self,m):
        """

        Samples m samples from the current NakaRushton distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        s = self.param['s']*2.0
        sampleDistribution = GammaP({'u':(self.param['n']/self.param['p']),'s':s,'p':self.param['p']})
        if self.param['gamma'] == 2.0:
            prop = sampleDistribution.cdf(Data(array([self.param['kappa']])))
            m0 = 0
            ret = sampleDistribution.sample(2*m*int(1.0/prop))
            ret.X = ret.X[where(ret.X <= self.param['kappa'])]
            ret.X = reshape(ret.X,(1,len(ret.X)))
            m0 = ret.numex()
            while m0 < m:
                tmp = sampleDistribution.sample(2*m*int(1.0/prop))
                tmp.X = ret.X[where(tmp.X <= self.param['kappa'])]
                tmp.X.reshape((1,len(tmp.X)))
                m0 += tmp.numex()
                ret.X = hstack((ret.X,tmp.X))
            ret = ret[0,:m]
            ret.X = ret.X*self.param['sigma'] / sqrt(self.param['kappa']**2.0 - ret.X**2.0)
        else:
            ret = sampleDistribution.sample(m)
            X = ret.X.flatten()
            f = lambda x: x*self.param['kappa']/sqrt(self.param['sigma']**2.0 + x**self.param['gamma'])
            tmp = invertMonotonicIncreasingFunction(f,X,0.0*X,2.0*X)
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
        kappa = self.param['kappa']
        n = self.param['n']
        p = self.param['p']
        s = self.param['s']
        sigma = self.param['sigma']
        gamma = self.param['gamma']
        r = squeeze(dat.X)

        if gamma == 2.0:
            ll = log(p) + n*log(kappa) + 2.0*log(sigma) + (n-1)*log(r) - log(gammainc(n/p,kappa**p/2.0/s)) \
                 -gammaln(n/p) - n/p*log(2*s) - (n+2.0)/2.0 * log(sigma**2 + r**2) - r**p*kappa**p/2.0/s/(sigma**2+r**2)**(p/2.0)
        else:
            ll = log(p) + n*log(kappa) + (n-1.0)*log(r) + log(2.0*sigma**2.0 + (2.0-gamma)* r**gamma) \
                 -log(2.0) -gammaln(n/p) - n/p*log(2*s) - (n+2.0)/2.0 * log(sigma**2 + r**gamma) \
                 - r**p * kappa**p / 2.0 /s /(sigma**2+r**gamma)**(p/2.0)
 
        return ll

    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''
        if self.param['gamma'] == 2.0:
            datf = dat.copy()
            datf.X = self.param['kappa'] * datf.X / sqrt(self.param['sigma']**2.0 + sum(datf.X**2,axis=0))
            tmp = GammaP(u=(self.param['n']/self.param['p']),s=2*self.param['s'],p=self.param['p'])
            return tmp.cdf(datf)/tmp.cdf(Data(array([[self.param['kappa']]])))
        else:
            raise NotImplementedError('NakaRushton.cdf is not implemented for gamma != 2.0!')
        
    def dldtheta(self,dat):
        """
        Evaluates the gradient of the NakaRushton loglikelihood with respect to the primary parameters.

        :param data: Data on which the gradient should be evaluated.
        :type data: DataModule.Data
        
        """
        
        m = dat.numex()
        grad = zeros((len(self.primary),m))
        kappa = self.param['kappa']
        n = float(self.param['n'])
        p = float(self.param['p'])
        s = float(self.param['s'])
        sigma = self.param['sigma']
        gamma = self.param['gamma']
        ind = 0
        r = dat.X
        
        for param in self.primary:
            if param == 'kappa':
                grad[ind,:] = n/kappa - p*kappa**(p-1.0)*r**p / 2.0/s/(sigma**2.0 + r**gamma)**(p/2.0)
            if param == 'sigma':
                grad[ind,:] = 4.0*sigma/(2.0*sigma**2.0 + (2-gamma)*r**gamma) \
                              - (n+2)*sigma/(sigma**2.0 + r**gamma) \
                              + p*kappa**p*r**p*sigma/(2.0*s*(sigma**2.0+r**gamma)**((p+2.0)/2.0))
            if param == 'gamma':
                grad[ind,:] =  r**gamma* ((2.0-gamma)*log(r)-1.0)  / (2*sigma**2.0 + (2-gamma)*r**gamma) \
                              - (n+2)*r**gamma*log(r) / 2.0 /(sigma**2.0 + r**gamma)\
                              + p * kappa**p * r**(p+gamma) * log(r) / 4.0 / s / (sigma**2.0 + r**gamma)**((p+2)/2)
            
            ind += 1
                
            
   
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
            if key == 'gamma':
                ret.append((1e-6,1.99))
            else:
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
            

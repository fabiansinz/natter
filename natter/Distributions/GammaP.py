from Gamma  import  Gamma
from natter.DataModule import Data
from numpy import log, exp, mean
from natter import Auxiliary
from scipy.stats import gamma


class GammaP(Gamma):

    def __init__(self,param=None):
        '''
        GammaP distribution constructor.
        
        :param param: Initial parameters for the GammaP distribution. The GammaP distribution has parameters *u* (shape parameter), *s* (scale parameter) and *p* (exponent). The default value for param is {'u':1.0,'s':1.0,'p':2.0}.
        :type param: dict.
        :returns:  A GammaP distribution object initialized with the parameters in param.

        Primary parameters are ['u','s','p'].

        
        '''
        self.name = 'GammaP Distribution'
        self.param = {'u':1.0, 'p':2.0, 's':1.0}
        if param!=None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['p','s','u']
        
    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        return Gamma.loglik(self,dat**self.param['p']) + log(self.param['p']) + (self.param['p']-1)*log(dat.X) 

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the derivatives.
        :rtype:    numpy.array
        
        """        
        return Gamma.dldx(self,dat**self.param['p'])*self.param['p']*dat.X**(self.param['p']-1.0) \
               +(self.param['p']-1)/dat.X
    

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
        return gamma.cdf(dat.X**self.param['p'],self.param['u'],scale=self.param['s'])

    def ppf(self,U):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param X: Percentiles for which the ppf will be computed.
        :type X: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        return Data(gamma.ppf(U,self.param['u'],scale=self.param['s'])**(1/self.param['p']))


    def sample(self,m):
        """

        Samples m samples from the current GammaP distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """

        dat = (Gamma.sample(self,m))**(1/self.param['p'])
        dat.setHistory([])
        return dat

    def estimate(self,dat,prange=(.1,5.0)):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).

        Estimate fits a Gamma distribution on :math:`x^p`.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param prange: Range to be search in for the optimal *p*.
        :type prange:  tuple
        
        '''

        if 'p' in self.primary:
            f = lambda t: self.__pALL(t,dat)
            bestp = Auxiliary.Optimization.goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])
        Gamma.estimate(self,dat**self.param['p'])

    def __pALL(self,p,dat):
        self.param['p'] = p
        pold = list(self.primary)
        pr= list(pold)
        if 'p' in pr:
            pr.remove('p')
        self.primary = pr
        self.estimate(dat)
        self.primary = pold
        return -mean(self.loglik(dat))





from __future__ import division
from LpSphericallySymmetric import LpSphericallySymmetric
from GammaP import GammaP
from natter.DataModule import Data
from numpy import mean, sum, abs, sign
from numpy.random import gamma, randn
from natter.Auxiliary.Optimization import goldenMinSearch
from copy import deepcopy

from natter.Auxiliary.Utils import parseParameters
from scipy import special
from Distribution import Distribution

class ChiP(Distribution):
    """
    ChiP Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    The ChiP distribution is the generalization of a Chi
    distribution (the radial distribution of a Gaussian). The ChiP
    distribution is the radial distribution of a p-Generalized
    Normal distribution. IMPORTANT: ChiP with p=2.0 is a Chi
    distribution and not a Chi^2.

    :param param: dictionary which might containt parameters for the Gamma distribution
        'n'    :    Degrees of freedom (the dimensionality of the p-generalized Normal)

        'p'    :    Exponent (default p=2.0 yields a Chi distribution)

        's'    :    Scale parameter (default = (gamma(1.0/p)/gamma(3.0/p))**(p/2.0) )

    :type param: dict

    Primary parameters are ['s'].
    """

    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        # (special.gamma(1.0/p)/special.gamma(3.0/p))**(p/2.0)
        param = parseParameters(args,kwargs)

        # set default parameters
        self.param = {'n':2, 'p':2.0}
        self.name = 'ChiP Distribution'
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        if not self.param.has_key('s'):
            self.param['s'] =  (special.gamma(1.0/self.param['p'])/special.gamma(3.0/self.param['p']))**(self.param['p']/2.0)
        self.primary = ['s']
        self.baseDist = GammaP({'u':(self.param['n']/float(self.param['p'])),'s':self.param['s'],'p':self.param['p']})


    def estimate(self,dat):
        '''
        Estimates the parameters from the data in dat.

        :param dat: Data for estimating self
        :type dat: natter.DataModule.Data

        '''

        self.param['s'] = self.param['p']*mean(sum(abs(dat.X)**self.param['p'],0))  / self.param['n']
        self.baseDist['s'] = self.param['s']

    def loglik(self,dat):
        """

        Computes the loglikelihood of the data points in dat.

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
        """
        return self.baseDist.loglik(dat)

    def __setitem__(self,key,value):
        if key in self.parameters('keys'):
            if key == 's':
                self.baseDist['s'] = value
            if key == 'p':
                self.baseDist['u'] = self.param['n']/value
            self.param[key] = value
        else:
            raise KeyError("Parameter %s not defined for %s" % (key,self.name))


    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution function on the data points in dat.

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array

        '''
        return self.baseDist.cdf(dat)


    def ppf(self,U):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param U: Percentiles for which the ppf will be computed.
        :type U: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data

        '''

        return self.baseDist.ppf(U)

    def sample(self,m):
        '''
           Samples m examples from the distribution.

           :param m: number of patches to sample
           :type m: int
           :returns: Samples from the ChiP distribution
           :rtype: natter.DataModule.Data

        '''
        dat = self.baseDist.sample(m)
        dat.name = dat.name.replace('GammaP','ChiP')
        return dat



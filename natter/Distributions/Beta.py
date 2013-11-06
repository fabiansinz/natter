from natter.Auxiliary.Utils import parseParameters
from scipy import stats
from scipy.special import digamma
from natter.Distributions import Distribution
from natter.DataModule import Data
from numpy import squeeze, zeros, log


class Beta(Distribution):
    """
    Beta Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    The Beta distribution is defined on [0,1].

    :param param: dictionary which might containt parameters for the Gamma distribution
        'alpha':    Shape parameter alpha

        'beta' :    Shape parameter beta

    :type param: dict

    Primary parameters are ['alpha','beta'].
    """


    def __init__(self, *args, **kwargs):
        param = parseParameters(args, kwargs)

        # set default parameters
        self.name = 'Beta Distribution'
        self.param = {'alpha': 1.0, 'beta': 1.0}

        if param is not None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['alpha', 'beta']

    def pdf(self, dat):
        '''

        Evaluates the probability density function on the data points in dat.

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array

        '''
        return squeeze(stats.beta.pdf(dat.X, self['alpha'], self['beta']))

    def loglik(self, dat):
        """

        Computes the loglikelihood of the data points in dat.

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
        """
        return log(self.pdf(dat))

    def sample(self, m):
        '''
        Samples m examples from the distribution.

        :param m: number of patches to sample
        :type m: int
        :returns: Samples from the ChiP distribution
        :rtype: natter.DataModule.Data

        '''
        return Data(stats.beta.rvs(self['alpha'], self['beta'], size=(m,)))

    def primary2array(self):
        """
        :returns: array containing primary parameters. If 'sigma' is in the primary parameters, then the cholesky factor of the precision matrix is filled in the array.
        """
        ret = zeros(len(self.primary))
        for ind, key in enumerate(self.primary):
            ret[ind] = self.param[key]
        return ret

    def array2primary(self, arr):
        """
        Takes an array with values for the primary parameters and stores them in the object.

        :param arr: array with primary parameters
        :type arr: numpy.ndarray

        """
        ind = 0
        for ind, key in enumerate(self.primary):
            self.param[key] = arr[ind]
        return self

    def primaryBounds(self):
        """
        Returns bound on the primary parameters.

        :returns: bound on the primary parameters
        :rtype: list of tuples containing the specific lower and upper bound
        """
        return len(self.primary) * [(1e-6, None)]


    def dldtheta(self, dat):
        """
        Evaluates the gradient of the distribution with respect to the primary parameters.

        :param dat: Data on which the gradient should be evaluated.
        :type dat: DataModule.Data
        :returns:   The gradient
        :rtype:     numpy.array

        """
        ret = zeros((len(self.primary), dat.numex()))
        x = dat.X[0]
        a = self['alpha']
        b = self['beta']
        p = self.pdf(dat)
        for ind, key in enumerate(self.primary):
            if key is 'alpha':
                ret[ind, :] = p * (digamma(a + b) - digamma(a) + log(x))
            elif key is 'beta':
                ret[ind, :] = p * (digamma(a + b) - digamma(b) + log(1 - x))
        return ret

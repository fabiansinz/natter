from natter.Auxiliary.Utils import parseParameters
from scipy import stats
from scipy.special import digamma
from natter.Distributions import Distribution
from natter.DataModule import Data
from numpy import squeeze, zeros, log

class Beta(Distribution):


        def __init__(self, *args, **kwargs):
            param = parseParameters(args,kwargs)
        
            # set default parameters
            self.name = 'Beta Distribution'
            self.param = {'alpha':1.0,'beta':1.0}

            if param is not None:
                for k in param.keys():
                    self.param[k] = float(param[k])
            self.primary = ['alpha','beta']

        def pdf(self, dat):
            return squeeze(stats.beta.pdf(dat.X,self['alpha'], self['beta'] ))

        def loglik(self,dat):
            return log(self.pdf(dat))

        def sample(self,m):
            return Data(stats.beta.rvs(self['alpha'], self['beta'],size=(m,)))

        def primary2array(self):
            ret = zeros(len(self.primary))
            for ind,key in enumerate(self.primary):
                ret[ind]=self.param[key]
            return ret

        def array2primary(self, arr):
            ind = 0
            for ind, key in enumerate(self.primary):
                self.param[key] = arr[ind]
            return self

        def primaryBounds(self):
            return len(self.primary)*[(1e-6,None)]


        def dldtheta(self, dat):
            ret = zeros((len(self.primary), dat.numex()))
            x = dat.X[0]
            a = self['alpha']
            b = self['beta']
            p = self.pdf(dat)
            for ind, key in enumerate(self.primary):
                if key is 'alpha':
                    ret[ind,:] = p*(digamma(a+b)-digamma(a)+log(x))
                elif key is 'beta':
                    ret[ind,:] = p*(digamma(a+b)-digamma(b)+log(1-x))
            return ret

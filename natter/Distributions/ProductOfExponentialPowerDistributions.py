from Distribution import Distribution
from natter.DataModule import Data
from numpy import zeros
from ExponentialPower import ExponentialPower
import sys

class ProductOfExponentialPowerDistributions(Distribution):
    """
      Product of Exponential Power Distributions

      Parameters and their defaults are:
         n:    dimensionality (default=2)

         P: list of exponential power distribution objects (must have
            the dimension n)
         
    """
    
    def __init__(self,param=None):
        self.name = 'Product of Exponential Power Distributions'
        self.param = {'n':2,'P':None}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        if self.param['P'] != None:
            self.param['n'] = len(self.param['P'])
        else:
            self.param['P'] = [ExponentialPower() for i in range(self.param['n'])]
        self.primary = ['P']

    def sample(self,m):
        X = zeros((self.param['n'],m))
        for i in xrange(self.param['n']):
            X[i,:] = self.param['P'][i].sample(m).X
        return Data(X,str(m) + ' samples from a ' + self.name)

    def estimate(self,dat):
            
        print "Fitting Product of Exponential Power distributions ..."
        for i in xrange(self.param['n']):
            print "\r\tDistribution %d ...                 "  % (i,) ,
            sys.stdout.flush()
            self.param['P'][i].primary = ['p','s']
            self.param['P'][i].estimate(dat[i,:])
        print "[Done]"

        
    def loglik(self,dat):
        ret = zeros((1,dat.size(1)))
        for i in xrange(self.param['n']):
            ret = ret + self.param['P'][i].loglik(dat[i,:])
        return ret
        
    


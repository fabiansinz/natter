from Distribution import Distribution
from natter.DataModule import Data
from numpy import zeros
from ExponentialPower import ExponentialPower
import sys

class ProductOfExponentialPowerDistributions(Distribution):
    """
      Product of Exponential Power Distributions

    :param param:
        dictionary which may contain parameters for the product of Exponential Power distribution

         'n' :    dimensionality (default=2)

         'P' : list of exponential power distribution objects (must have
            the dimension n)
         
    :type param: dict

    Primary parameters are ['P'].
        
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
        """

        Samples m samples from the current productof  Exponential Power distributions.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples

        """
        
        X = zeros((self.param['n'],m))
        for i in xrange(self.param['n']):
            X[i,:] = self.param['P'][i].sample(m).X
        return Data(X,str(m) + ' samples from a ' + self.name)

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).


        :param dat: Data points on which the  ProductOfExponentialPowerDistributions will be estimated.
        :type dat: natter.DataModule.Data
        '''
            
        print "Fitting Product of Exponential Power distributions ..."
        for i in xrange(self.param['n']):
            print "\r\tDistribution %d ...                 "  % (i,) ,
            sys.stdout.flush()
            self.param['P'][i].primary = ['p','s']
            self.param['P'][i].estimate(dat[i,:])
        print "[Done]"

        
    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
           
        '''
        
        ret = zeros((1,dat.size(1)))
        for i in xrange(self.param['n']):
            ret = ret + self.param['P'][i].loglik(dat[i,:])
        return ret
        
    


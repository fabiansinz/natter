from Distribution import Distribution
from natter.DataModule import Data
from numpy import zeros, array, hstack
from ExponentialPower import ExponentialPower
import sys
from copy import deepcopy

class ProductOfExponentialPowerDistributions(Distribution):
    """
      Product of Exponential Power Distributions

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which may contain parameters for the product of Exponential Power distribution

         'n' :    dimensionality (default=2)

         'P' : list of exponential power distribution objects (must have
            the dimension n)
         
    :type param: dict

    Primary parameters are ['P'].
        
    """
    
    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = None
        if len(args) > 0:
            param = args[0]
        if kwargs.has_key('param'):
            if param == None:
                param = kwargs['param']
            else:
                for k,v in kwargs['param'].items():
                    param[k] = v
        if len(kwargs)>0:
            if param == None:
                param = kwargs
            else:
                for k,v in kwargs.items():
                    if k != 'param':
                        param[k] = v
        
        # set default parameters
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

    def parameters(self,keyval=None):
        """

        Returns the parameters of the distribution as dictionary. This
        dictionary can be used to initialize a new distribution of the
        same type. If *keyval* is set, only the keys or the values of
        this dictionary can be returned (see below). The keys can be
        used to find out which parameters can be accessed via the
        __getitem__ and __setitem__ methods.

        :param keyval: Indicates whether only the keys or the values of the parameter dictionary shall be returned. If keyval=='keys', then only the keys are returned, if keyval=='values' only the values are returned.
        :type keyval: string
        :returns:  A dictionary containing the parameters of the distribution. If keyval is set, a list is returned. 
        :rtype: dict or list
           
        """
        if keyval == None:
            return deepcopy(self.param)
        elif keyval== 'keys':
            return self.param.keys()
        elif keyval == 'values':
            return self.param.value()

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
        
    

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Product of Exponential Power distributions w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array
        
        """
        ret = zeros(dat.size())
        for i in xrange(self.param['n']):
            ret[i,:] = self.param['P'][i].dldx(dat[i,:])
        return ret
        
    def primary2array(self):
        ret = array([])
        if 'P' in self.primary:
            for p in self.param['P']:
                ret = hstack((ret,p.primary2array()))
        return ret

    def array2primary(self,ar):
        if len(ar) > 0:
            for i,p in enumerate(self.param['P']):
                l = len(p.primary2array())
                self.param['P'][i].array2primary(ar[:l])
                ar = ar[l:]
                

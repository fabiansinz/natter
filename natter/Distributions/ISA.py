from Distribution import Distribution
from LpSphericallySymmetric import LpSphericallySymmetric
from numpy import zeros
from natter.Auxiliary.Errors import InitializationError
from natter.DataModule import Data
import sys
from copy import deepcopy
 
class ISA(Distribution):
    """
    Independent Subspace Analysis (ISA)

    :param param:
        dictionary which may contain parameters for the product of the ISA

         'n' :    dimensionality 

         'S' :    list of index tuples corresponding to the indices into the subspaces (must be non-empty at initializaton)

         'P' :    list of Distribution objects corresponding to the distributions on those subspaces (must have the same length as 'S')
         
    :type param: dict

    Primary parameters are ['P'].
        
    """

    def __init__(self,param=None):
        self.name = 'Independent Subspace Analysis (ISA)'
        self.param = {'n':3,'P':None,'S':None}
        
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        if self.param['S'] == None:
            raise InitializationError('Parameter \'S\' must be non-empty!')

        if self.param['S'] != None:
            self.param['n'] = 0
            for k in xrange(len(self.param['S'])):
                self.param['n'] += len(self.param['S'][k])

        if self.param['P'] == None:
            self.param['P'] = [LpSphericallySymmetric({'n':len(elem)}) for elem in self.param['S']]
        self.primary = ['P']

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

    def sample(self,m):
        """

        Samples m samples from the current ISA model.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples

        """
        
        X = zeros((self.param['n'],m))
        S = self.param['S']
        P = self.param['P']
        for k in xrange(len(S)):
            X[S[k],:] = P[k].sample(m).X
        return Data(X,str(m) + ' samples from a ' + self.name)

    def estimate(self,dat):
        '''

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on
        the Distributions module <tutorial_Distributions>`).

        :param dat: Data points on which the ISA model will be estimated.
        :type dat: natter.DataModule.Data
        '''
            
        print "\tFitting ISA model ..."
        P = self.param['P']
        S = self.param['S']
        for i in xrange(len(P)):
            print "\r\tDistribution %d on %d-dimensional subspace ...  "  % (i,len(S[i])) ,
            sys.stdout.flush()
            P[i].estimate(dat[S[i],:])
        print "[Done]"

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the derivatives.
        :rtype:    numpy.array
        
        """        

        dlogdx = zeros(dat.size())
        P = self.param['P']
        S = self.param['S']
        for k in xrange(len(S)):
            dlogdx[S[k],:] = P[k].dldx(dat[S[k],:])

        return dlogdx

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        l = zeros((1,dat.size(1)))
        P = self.param['P']
        S = self.param['S']
        for k in xrange(len(S)):
            l = l + P[k].loglik(dat[S[k],:])
        return l 


       

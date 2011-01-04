from __future__ import division
from Distribution import Distribution
from numpy import  ones, eye,dot,log,sum,pi,where,zeros,outer,array,mean
from natter.Transforms import LinearTransform
from scipy.optimize import fmin_l_bfgs_b
from copy import deepcopy

class SchwartzSimoncelliModel(Distribution):
    """
    Schwartz Simoncelli Model

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the Gaussian
              'n'    :  dimensionality (default=2)
              
              'sigma':  offset for the covariance matrix
              
              'W' : Transforms.LinearTransform object that models the neighborhood function for the single variances. The diagonal of W is constrained to be 0 and the single entries are constrained to be positive. 

    :type param: dict

    Primary parameters are ['sigma','W'].
        
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
        self.name = 'Schwartz Simoncelli Model'
        self.param = {'n':2,'sigma':1.0}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        if not self.param.has_key('W'):
            self.param['W'] = LinearTransform(ones( (self.param['n'],self.param['n'])) - eye(self.param['n']),'Neighborhood function')
        self.primary = ['W','sigma']

        # indices into W that correspond to the offdiagonal term
        self.offDiagI,self.offDiagJ = where(ones( (self.param['n'],self.param['n'])) - eye(self.param['n']))

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

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on
        the Distributions module <tutorial_Distributions>`).

        The estimation is basically a bounded gradient descent on the
        ALL (average negative average log-likelihood per dimension in
        bits). *sigma* and *W* are both bounded away from zero and
        otherwise arbitrary.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        
        '''
        
        theta = self.primary2array()
        bounds = len(theta)*[(0.0,None)]

        theta = fmin_l_bfgs_b(self.__optfunc,theta,self.__optfuncprime,(dat,),bounds=bounds)[0]
        self.array2primary(theta)
        
        

    def __optfunc(self,theta,dat):
        theta0 = self.primary2array()
        self.array2primary(theta)
        r = self.all(dat)
        self.array2primary(theta0)
        return r

    def __optfuncprime(self,theta,dat):
        theta0 = self.primary2array()
        self.array2primary(theta)
        r = self.dldtheta(dat)
        self.array2primary(theta0)
        
        return -mean(r,1)/log(2)/dat.dim()

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat (Since
        the Schwartz Simoncelli model is not a true joint
        distribtions, this is not a true likelihood!).

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
           
        '''
        
        Y2 = dat.X**2.0
        W = self.param['W'].W
        myvar = dot(W,Y2) + self.param['sigma']**2
        return sum(-.5*log(2*pi) - .5*log(myvar) - Y2/2.0/myvar,0)


    def primary2array(self):
        """

        Returns the primary parameters in an array. The order is
        alphabetical: sigma first, then W.

        :returns: primary parameters in a numpy array
        :rtype: numpy.array
        
        """

        n = 0
        if 'sigma' in self.primary:
            n += 1
        if 'W' in self.primary: # all entries in W without the diagonal
            n += self.param['W'].W.shape[0]**2 - self.param['W'].W.shape[0]
            
        ret = zeros((n,))
        n = 0
        if 'sigma' in self.primary:
            ret[n] = self.param['sigma']
            n += 1
        if 'W' in self.primary: # all entries in W without the diagonal
            ret[n:] = self.param['W'].W[self.offDiagI,self.offDiagJ]
        
        return ret

    def array2primary(self,a):
        """

        Sets the primary parameters to the values stored in the
        array. The order is alphabetical: sigma first, then W.

        :param a: the primary parameter values
        :type a: numpy.array
        """

        n = 0
        if 'sigma' in self.primary:
            self.param['sigma'] = a[n]
            n += 1
        if 'W' in self.primary: # all entries in W without the diagonal
            self.param['W'].W[self.offDiagI,self.offDiagJ] = a[n:]


    def dldtheta(self,dat):
        """

        Computes the gradients of the ''log-likelihood'' of the model
        w.r.t. the primary parameters at the data points in dat.


        :param dat: Data points at which the gradient is to be computed
        :type dat: natter.DataModule.Data
        :returns: The gradients
        :rtype: numpy.array
        
        """

        Y2 = dat.X**2.0
        W = self.param['W'].W
        myvar = dot(W,Y2) + self.param['sigma']**2
        m = dat.numex()

        n = 0
        if 'sigma' in self.primary:
            n += 1
        if 'W' in self.primary: # all entries in W without the diagonal
            n += self.param['W'].W.shape[0]**2 - self.param['W'].W.shape[0]
            
        grad = zeros((n,m))
        


        k = 0
        if 'sigma' in self.primary:
            grad[k,:] = self.param['sigma']*sum((Y2-myvar)/myvar**2,0)
            k += 1
        if 'W' in self.primary:
            for i in xrange(m):
                grad[k:,i] = outer((Y2[:,i]-myvar[:,i])/myvar[:,i]**2,Y2[:,i]/2.0)[self.offDiagI,self.offDiagJ]
        return grad
            

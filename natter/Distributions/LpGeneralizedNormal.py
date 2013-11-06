from __future__ import division
from LpSphericallySymmetric import LpSphericallySymmetric
from GammaP import GammaP
from natter.DataModule import Data
from numpy import mean, sum, abs, sign
from numpy.random import gamma, randn
from natter.Auxiliary.Optimization import goldenMinSearch
from copy import deepcopy

class LpGeneralizedNormal(LpSphericallySymmetric):
    '''
      Lp-Generalized Normal Distribution

      The constructor is either called with a dictionary, holding
      the parameters (see below) or directly with the parameter
      assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
      also possible.

      Parameters and their defaults are:
         n:  dimensionality (default n=2)
         p:  exponent for the p-norm (default p=2.0)
         s:  scale (default s=1.0)
    '''

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
        self.param = {'n':2, 'p':2.0,'s':1.0}
        self.name = 'Lp-Generalized Normal Distribution'
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.param['rp'] = GammaP({'u':(self.param['n']/self.param['p']),'s':self.param['s'],'p':self.param['p']})
        self.primary = ['p','s']

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

    def estimate(self,dat,prange=(.1,5.0)):
        '''Estimates the parameters from the data in DAT. The optional
        second argument specifys a list of parameters that should be
        estimated. PRANGE, when specified, defines the search range for p.

        :param dat: data object with data points
        :type dat: natter.DataModule.Data
        :param prange: range of p-values to search in
        :type prange: tuple of floats
        '''
        if 'p' in self.primary:
            f = lambda t: self.__pALL(t,dat)
            bestp = goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])

        self.param['s'] = self.param['p']*mean(sum(abs(dat.X)**self.param['p'],0))  / self.param['n']
        self.param['rp'].param['s'] = self.param['s']
        self.param['rp'].param['u'] = float(self.param['n'])/self.param['p']

    def sample(self, m):

        '''
        Samples m examples from the distribution.

        :param m: number of patches to sample
        :type m: int
        :returns: Samples from the ChiP distribution
        :rtype: natter.DataModule.Data

        '''

        z = gamma(1 / self.param['p'], self.param['s'], (self.param['n'], m))
        z = abs(z) ** (1 / self.param['p'])
        return Data(z * sign(randn(self.param['n'], m)), 'Samples from ' + self.name, \
                         ['sampled ' + str(m) + ' examples from Lp-generalized Normal'])


    def __setitem__(self,key,value):
        if key in self.parameters('keys'):
            if key == 's':
                self.param['s'] = value
                self.param['rp'].param['s'] = value
            elif key == 'p':
                self.param['p'] = value
                self.param['rp'].param['p'] = value
                self.param['rp'].param['u'] = float(self.param['n'])/self.param['p']
            else:
                self.param[key] = value
        else:
            raise KeyError("Parameter %s not defined for %s" % (key,self.name))

    def __pALL(self,p,dat):
        self.param['p'] = p
        self.param['rp'].param['p'] = p
        pold = list(self.primary)
        pr = list(pold)
        if 'p' in pr:
            pr.remove('p')
        self.primary = pr
        self.estimate(dat)
        self.primary = pold
        return self.all(dat)

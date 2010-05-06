import Distribution
import Data
import numpy as np
import scipy as sp
from scipy import special
import GammaP
import LpSphericallySymmetric
import Auxiliary

class LpGeneralizedNormal(LpSphericallySymmetric.LpSphericallySymmetric):
    '''
      Lp-Generalized Normal Distribution

      Parameters and their defaults are:
         n:  dimensionality (default n=2)
         p:  exponent for the p-norm (default p=2.0)
         s:  scale (default s=1.0)
    '''

    def __init__(self,param=None):
        self.param = {'n':2, 'p':2.0,'s':1.0}
        self.name = 'Lp-Generalized Normal Distribution'
        if param != None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.param['rp'] = GammaP.GammaP({'u':(self.param['n']/self.param['p']),'s':self.param['s'],'p':self.param['p']})

    def estimate(self,dat,which=None,prange=(.1,5.0)):
        '''ESTIMATE(DAT[, WHICH=SELF.PARAM.KEYS(),[PRANGE=(.1,5.0)]])
        
        estimates the parameters from the data in DAT. The optional
        second argument specifys a list of parameters that should be
        estimated. PRANGE, when specified, defines the search range for p.
        '''
        if which == None:
            which = self.param.keys()
        if which.count('p') > 0:
            f = lambda t: self.__pALL(t,dat)
            bestp = Auxiliary.Optimization.goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])

        self.param['s'] = self.param['p']*np.mean(np.sum(np.abs(dat.X)**self.param['p'],0))  / self.param['n']
        self.param['rp'].param['s'] = self.param['s']
        self.param['rp'].param['u'] = self.param['n']/self.param['p']
        
    def sample(self,m):
        '''SAMPLE(M)

           samples M examples from the distribution.
        '''
        z = np.random.gamma(1/self.param['p'],self.param['s'],(self.param['n'],m))
        z = np.abs(z)**(1/self.param['p'])
        return Data.Data(z * np.sign(np.random.randn(self.param['n'],m)),'Samples from ' + self.name, \
                      ['sampled ' + str(m) + ' examples from Lp-generalized Normal'])

    def __pALL(self,p,dat):
        self.param['p'] = p
        self.param['rp'].param['p'] = p
        self.estimate(dat,['s'])
        return self.all(dat)

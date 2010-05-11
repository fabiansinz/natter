import Distributions
from DataModule import Data
from numpy import mean, sum, abs, sign
from numpy.random import gamma, randn
import Auxiliary

class LpGeneralizedNormal(Distributions.LpSphericallySymmetric):
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
        self.param['rp'] = Distributions.GammaP({'u':(self.param['n']/self.param['p']),'s':self.param['s'],'p':self.param['p']})
        self.primary = ['p','s']

    def estimate(self,dat,prange=(.1,5.0)):
        '''ESTIMATE(DAT,[PRANGE=(.1,5.0)]])
        
        estimates the parameters from the data in DAT. The optional
        second argument specifys a list of parameters that should be
        estimated. PRANGE, when specified, defines the search range for p.
        '''
        if 'p' in self.primary:
            f = lambda t: self.__pALL(t,dat)
            bestp = Auxiliary.Optimization.goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])

        self.param['s'] = self.param['p']*mean(sum(abs(dat.X)**self.param['p'],0))  / self.param['n']
        self.param['rp'].param['s'] = self.param['s']
        self.param['rp'].param['u'] = self.param['n']/self.param['p']
        
    def sample(self,m):
        '''SAMPLE(M)

           samples M examples from the distribution.
        '''
        z = gamma(1/self.param['p'],self.param['s'],(self.param['n'],m))
        z = abs(z)**(1/self.param['p'])
        return Data(z * sign(randn(self.param['n'],m)),'Samples from ' + self.name, \
                      ['sampled ' + str(m) + ' examples from Lp-generalized Normal'])

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

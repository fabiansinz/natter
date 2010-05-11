from Distributions import Distribution, Gamma
from DataModule import Data
from numpy import log, abs, sign
from numpy.random import gamma, randn
from scipy.special import gammaln
import Auxiliary

class LpSphericallySymmetric(Distribution):
    '''
      Lp-Spherically Symmetric Distribution

      Parameters and their defaults are:
         n:  dimensionality (default n=2)
         rp: radial distribution (default rp=Gamma())
         p:  exponent for the p-norm (default p=2)
    '''

    def __init__(self,param=None):

        self.name = 'Lp-Spherically Symmetric Distribution'
        self.param = {'n':2, 'rp':Gamma(),'p':2.0}
        if param != None: 
            for k in param.keys():
                self.param[k] = param[k]
        self.param['p'] = float(self.param['p']) # make sure it is a float
        self.prange = (.1,2.0)
        self.primary = ['rp','p']

    def logSurfacePSphere(self):
        return self.param['n']*log(2) + self.param['n']*gammaln(1/self.param['p']) \
               - gammaln(self.param['n']/self.param['p']) - (self.param['n']-1)*log(self.param['p']);

    def loglik(self,dat):
        r = dat.norm(self.param['p'])
        return self.param['rp'].loglik(r) \
               - self.logSurfacePSphere() - (self.param['n']-1)*log(r.X)

    def dldx(self,dat):
        """
        DLDX(DAT)

        returns the derivative of the log-likelihood w.r.t. the data points in DAT.
        """
        drdx = dat.dnormdx(self.param['p'])
        r = dat.norm(self.param['p'])
        tmp = (self.param['rp'].dldx(r) - (self.param['n']-1.0)*1.0/r.X)
        for k in range(len(drdx)):
            drdx[k] *= tmp
        return drdx
        
    def estimate(self,dat,prange=None):
        '''ESTIMATE(DAT[, [PRANGE=(.1,5.0)]])
        
        estimates the parameters from the data in DAT. PRANGE, when
        specified, defines the search range for p.
        '''
        if not prange:
            prange = self.prange
        if 'p' in self.primary:
            f = lambda t: self.__pALL(t,dat)
            bestp = Auxiliary.Optimization.goldenMinSearch(f,prange[0],prange[1],5e-4)
            self.param['p'] = .5*(bestp[0]+bestp[1])
        if 'rp' in self.primary:
            self.param['rp'].estimate(dat.norm(self.param['p']))
        self.prange = (self.param['p']-.5,self.param['p']+.5)
            
    def __pALL(self,p,dat):
        self.param['rp'].estimate(dat.norm(p))
        self.param['p'] = p
        return self.all(dat)
        
    def sample(self,m):
        '''SAMPLE(M)

           samples M examples from the distribution.
        '''
        # sample from a p-generlized normal with scale 1
        z = gamma(1/self.param['p'],1.0,(self.param['n'],m))
        z = abs(z)**(1/self.param['p'])
        dat =  Data(z * sign(randn(self.param['n'],m)),'Samples from ' + self.name, \
                      ['sampled ' + str(m) + ' examples from Lp-generalized Normal'])
        # normalize the samples to get a uniform distribution.
        dat.normalize(self.param['p'])
        r = self.param['rp'].sample(m)
        dat.scale(r)
        return dat


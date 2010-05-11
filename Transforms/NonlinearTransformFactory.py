import numpy as np
from NonlinearTransform import NonlinearTransform
from numpy import linalg
from scipy import special
#from  Distributions import GammaP,LpSphericallySymmetric, LpGeneralizedNormal
import Distributions
from Auxiliary import Errors
import types

################################ NONLINEAR FILTERS ########################################

def RadialFactorization(psource):
    if not isinstance(psource,LpSphericallySymmetric):
        raise TypeError('Transform.TransformFactory.RadialFactorization: psource must be a Lp-spherically symmetric distribution')
    ptarget = Distributions.LpGeneralizedNormal({'s':(special.gamma(1.0/psource.param['p'])/special.gamma(3.0/psource.param['p']))**(psource.param['p']/2.0),\
                                                'p':psource.param['p']})
    print ptarget
    return RadialTransformation(psource,ptarget)

def RadialTransformation(psource,ptarget):
    if ptarget.param['p'] != psource.param['p']:
        raise Errors.ValueError('Values of p must agree')
    p = ptarget.param['p']
    name = 'Radial Factorization Transform ' + psource.name + ' (' + psource.param['rp'].name + ') --> ' + ptarget.name + ' (' + ptarget.param['rp'].name + ')'
    psource = psource.param['rp'].copy()
    ptarget = ptarget.param['rp'].copy()
    g = lambda x: x.scaleCopy( ptarget.ppf(psource.cdf(x.norm(p))).X / x.norm(p).X)
    gdet = lambda y: logDetJacobianRadialTransform(y,psource,ptarget,p)
    return NonlinearTransform(g,name,logdetJ=gdet)

def logDetJacobianRadialTransform(dat,psource,ptarget,p):
    r = dat.norm(p)
    r2 = ptarget.ppf(psource.cdf(r))
    n = dat.size(0)
    return (n-1)*np.log(r2.X) - (n-1)*np.log(r.X) + psource.loglik(r) - ptarget.loglik(r2)
    
#################################################

def LpNestedNonLinearICA(p):
    L = p.param['f'].copy()
    rp = p.param['rp'].copy()

    return getLpNestedNonLinearICARec(rp,L,())
    
def getLpNestedNonLinearICARec(rp,L,mind):
    p = L.p[L.pdict[()]]
    s = (special.gamma(1.0/p) / special.gamma(3.0/p))**(p/2.0)
    #    s = rp.param['s']
    rptarget = Distributions.GammaP({'u':float(L.n[()])/L.p[L.pdict[()]],'s':s, 'p':L.p[L.pdict[()]]})
    g = lambda x: x.scaleCopy( rptarget.ppf(rp.cdf(L(x))).X / L(x).X, L.iByI[()])
    gdet = lambda y: logDetJacobianLpNestedTransform(y,rp,rptarget,L)
    F = NonlinearTransform(g,'Rescaling of ' + str(mind),logdetJ=gdet)

    for k in range(L.l[()]):
        if L.n[(k,)] > 1:
            L2 = L[(k,)]
            pnew = Distributions.GammaP({'u':float(L2.n[()])/L.p[L.pdict[()]],'s':s, 'p':L.p[L.pdict[()]]})
            F2 = getLpNestedNonLinearICARec(pnew,L2,mind + (k,))
            F = F2*F


    return F
    
def logDetJacobianLpNestedTransform(dat,psource,ptarget,L):
    r = L(dat)
    r2 = ptarget.ppf(psource.cdf(r))
    n = L.n[()]
    return (n-1)*np.log(r2.X) - (n-1)*np.log(r.X) + psource.loglik(r) - ptarget.loglik(r2)

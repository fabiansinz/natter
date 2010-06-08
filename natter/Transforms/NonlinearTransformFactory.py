from numpy import log, abs
from natter.Distributions import GammaP,LpSphericallySymmetric, LpGeneralizedNormal
from NonlinearTransform import NonlinearTransform
from scipy import special
from natter.Auxiliary import Errors


################################ NONLINEAR FILTERS ########################################

def RadialFactorization(psource):
    """
    Creates a non-linear transformation that maps a radial
    distribution of any Lp-spherically symmetric distribution into the
    radial distribution of a Lp-generalized Normal distribution. The
    resulting filter acts on samples from the original
    distribution. For further details see ([SinzBethge2009]_).

    :param psource: Source distribution which must be Lp-spherically symmetric
    :type psource: natter.Distributions.LpSphericallySymmetric
    :returns: A non-linear filter that maps samples from psource into samples from a Lp-generalized Normal.
    :rtype: natter.Transforms.NonlinearTransform
    
    """
    if not isinstance(psource,LpSphericallySymmetric):
        raise TypeError('Transform.TransformFactory.RadialFactorization: psource must be a Lp-spherically symmetric distribution')
    ptarget = LpGeneralizedNormal({'s':(special.gamma(1.0/psource.param['p'])/special.gamma(3.0/psource.param['p']))**(psource.param['p']/2.0), 'p':psource.param['p']})
    return RadialTransformation(psource,ptarget)

def RadialTransformation(psource,ptarget):
    """
    Creates a non-linear transform that maps samples from one
    Lp-spherically symmetric distribution into that of another by
    histogram equalization on the radial component (see [SinzBethge2009]_).

    :param psource: Source distribution which must be Lp-spherically symmetric
    :type psource: natter.Distributions.LpSphericallySymmetric
    :param ptarget: Target distribution which must be Lp-spherically symmetric
    :type ptarget: natter.Distributions.LpSphericallySymmetric
    :returns: A non-linear filter that maps samples from psource into samples from ptarget.
    :rtype: natter.Transforms.NonlinearTransform
    
    
    """
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
    """
    Computes the log-determinant of the transformation between two
    Lp-spherically symmetric distributions (see additional material of
    [SinzBethge2009]_).

    :param dat: Points at which the log-determinant is to be computed.
    :type dat: natter.DataModule.Data
    :param psource: Source distribution which must be Lp-spherically symmetric
    :type psource: natter.Distributions.LpSphericallySymmetric
    :param ptarget: Target distribution which must be Lp-spherically symmetric
    :type ptarget: natter.Distributions.LpSphericallySymmetric
    :param p: p of the Lp-norm
    :type p: float
    :returns: log-determinant
    :rtype: float
    
    """
    r = dat.norm(p)
    r2 = ptarget.ppf(psource.cdf(r))
    n = dat.size(0)
    return (n-1)*log(r2.X) - (n-1)*log(r.X) + psource.loglik(r) - ptarget.loglik(r2)

    

#################################################

def LpNestedNonLinearICA(p):
    """
    Creates a non-linera filter that maps the samples from the
    Lp-nested symmetric distribution *p* into samples from a Product
    of Exponential Power distributions (see [SinzBethge2010]_).

    :param p: Lp-nested symmetric source distribution.
    :type p: natter.Distributions.LpNestedSymmetric
    :returns: non-linear transform
    :rtype: natter.Transforms.NonlinearTransform
    
    """
    L = p.param['f'].copy()
    rp = p.param['rp'].copy()

    return getLpNestedNonLinearICARec(rp,L,())
    
def getLpNestedNonLinearICARec(rp,L,mind):
    p = L.p[L.pdict[()]]
    s = (special.gamma(1.0/p) / special.gamma(3.0/p))**(p/2.0)
    rptarget = GammaP({'u':float(L.n[()])/L.p[L.pdict[()]],'s':s, 'p':L.p[L.pdict[()]]})
    g = lambda x: x.scaleCopy( rptarget.ppf(rp.cdf(L(x))).X / L(x).X, L.iByI[()])
    gdet = lambda y: logDetJacobianLpNestedTransform(y,rp,rptarget,L)
    F = NonlinearTransform(g,'Rescaling of ' + str(mind),logdetJ=gdet)

    for k in range(L.l[()]):
        if L.n[(k,)] > 1:
            L2 = L[(k,)]
            pnew = GammaP({'u':float(L2.n[()])/L.p[L.pdict[()]],'s':s, 'p':L.p[L.pdict[()]]})
            F2 = getLpNestedNonLinearICARec(pnew,L2,mind + (k,))
            F = F2*F


    return F
    
def logDetJacobianLpNestedTransform(dat,psource,ptarget,L):
    """
    Computes the log-determinant of the transformation between two
    Lp-nested symmetric distributions in the Lp-nested symmetric ICA
    (as described in [SinzBethge2010]_). Note that the non-linear
    filter created with *LpNestedNonLinearICA* provides a
    log-determinant.

    :param dat: Points at which the log-determinant is to be computed.
    :type dat: natter.DataModule.Data
    :param psource: Source distribution
    :type psource: natter.Distributions.LpSphericallySymmetric
    :param ptarget: Target distribution
    :type ptarget: natter.Distributions.LpNestedSymmetric
    :param L: Lp-nested function
    :type L: natter.Auxiliary.LpNestedFunction
    :returns: log-determinant
    :rtype: float
    
    """
    r = L(dat)
    r2 = ptarget.ppf(psource.cdf(r))
    n = L.n[()]
    return (n-1)*log(r2.X) - (n-1)*log(r.X) + psource.loglik(r) - ptarget.loglik(r2)


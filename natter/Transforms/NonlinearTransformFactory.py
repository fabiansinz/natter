from numpy import log, squeeze, mod, eye
from natter.Distributions import GammaP,LpSphericallySymmetric, LpGeneralizedNormal
from NonlinearTransform import NonlinearTransform
from LinearTransform import LinearTransform
from LinearTransformFactory import SSA
from scipy import special
from natter.Auxiliary import Errors


################################ NONLINEAR FILTERS ########################################

def ElementWise(g):
    """
    Creates a non-linear filter that applies the function g to each
    dimension of the data the filter is applied to. g must take a numpy.array and return a numpy array.

    
    
    :param g: Function handle that specifies the tranformation on each dimension. 
    :type g: function handle
    :returns: A non-linear filter that changes applies g dimensionwise
    :rtype: natter.Transforms.NonlinearTransform
    
    """
    def tr(dat):
        ret = dat.copy()
        for k in xrange(ret.dim()):
            ret.X[k,:] = g(ret.X[k,:])
        return ret
    return NonlinearTransform(tr,"Elementwise non-linear transformation.")

def MarginalHistogramEqualization(psource,ptarget=None):
    """
    Creates a non-linear filter that changes the marginal distribution
    of each single data dimension independently. For that sake it
    takes two ISA models and performs a histogram equalization on each
    of the marginal distributions.

    *Important*: The ISA models must have one-dimensional subspaces!

    If ptarget is omitted, it will be set to a N(0,I) Gaussian by default.
    
    :param psource: Source distribution which must be a natter.Distributions.ISA model with one-dimensional subspaces
    :type psource: natter.Distributions.ISA
    :param ptarget: Target distribution which must be a natter.Distributions.ISA model with one-dimensional subspaces
    :type ptarget: natter.Distributions.ISA
    :returns: A non-linear filter that changes for marginal distributions of the data from the respective psource into the respective ptarget
    :rtype: natter.Transforms.NonlinearTransform
    
    """
    from natter.Distributions import ISA, Gaussian

    if not isinstance(psource,ISA):
        raise TypeError('Transform.TransformFactory.MarginalHistogramEqualization: psource must be an ISA model')
    else:
        psource = psource.copy()
        
    if not ptarget == None and not isinstance(ptarget,ISA):
        raise TypeError('Transform.TransformFactory.MarginalHistogramEqualization: ptarget must be an ISA model')
    

    for ss in psource['S']:
        if len(ss) != 1:
            raise Errors.DimensionalityError('Transform.TransformFactory.MarginalHistogramEqualization: psource must have one-dimensional subspaces')

    if ptarget == None:
        ptarget = ISA(S=[(k,) for k in range(psource['n'])],P=[Gaussian(n=1) for k in range(psource['n'])])
    else:
        ptarget = ptarget.copy()


    g = lambda dat: reduce(lambda x,y: x.stack(y),[ptarget['P'][k].ppf(psource['P'][k].cdf(dat[k,:])) for k in range(psource['n']) ]  )
    gdet = lambda y: psource.loglik(y) - ptarget.loglik(g(y))

    name = 'Marginal Histogram Equalization Transform: %s --> %s' % (psource['P'][0].name, ptarget['P'][0].name)
    return NonlinearTransform(g,name,logdetJ=gdet)
    

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
    p = psource.param['p']
    ptarget = LpGeneralizedNormal({'n':psource['n'],'s':(special.gamma(1.0/p)/special.gamma(3.0/p))**(p/2.0), 'p':p})
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
        raise ValueError('Values of p must agree')
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
    return squeeze((n-1)*log(r2.X) - (n-1)*log(r.X) + psource.loglik(r) - ptarget.loglik(r2))

    

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

    return _getLpNestedNonLinearICARec(rp,L,())
    
def _getLpNestedNonLinearICARec(rp,L,mind):
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
            F2 = _getLpNestedNonLinearICARec(pnew,L2,mind + (k,))
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
    return squeeze((n-1)*log(r2.X) - (n-1)*log(r.X) + psource.loglik(r) - ptarget.loglik(r2))


def SSA2D( linearfilter=None, data=None, *args, **kwargs ):
    """
    Creates a nonlinear filter either from the given linear SSA filter
    or learns the linear filter on given data set using the
    LinearTransformFactory.SSA() method. The SSA2D filter computes the
    sum of the squared responses of 1st and 2nd, 3rd and 4th, ...
    component thus returns n/2 dimensions.
    
    :param linearfilter: the linear filter stage of the nonlinear filter
    :type linearfilter: natter.Transforms.LinearTransform
    :param data: Alternatively data on which the linear filter is learned
    :type data: natter.DataModule.Data

    :returns: bib-linear transform
    :rtype: natter.Transforms.NonlinearTransform
    
    """
    if linearfilter is None and not data is None:
        U = SSA(data, *args, **kwargs)
    elif not linearfilter is None:
        U = linearfilter
    else:
        raise ValueError('in NonlinearTransformFactory.SSA2D both linearfilter and data cannot be None')

    if mod(U.W.shape[0], 2) == 1:
        raise Errors.DimensionalityError('Transform must have even dimension number')
        
    g = ElementWise( lambda x: x**2 )
    g.name = 'Elementwise squaring'
    M = LinearTransform( eye(U.W.shape[0]).reshape(U.W.shape[0]//2, 2, U.W.shape[0]).sum(1), name='Summing over 2D subspaces' )
    nonlinearfilter = M*g*U
    nonlinearfilter.name = '2D SSA filter'
    
    return nonlinearfilter
    

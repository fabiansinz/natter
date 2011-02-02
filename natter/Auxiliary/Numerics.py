from __future__ import division
from scipy.maxentropy import maxentutils 
from numpy import asarray, log, exp, array, where, float64, shape, reshape,  pi, min,max, ndarray, zeros
from scipy import special
import types
from Errors import DimensionalityError
from scipy.integrate import quad

def owensT(h,a):
    """
    Owen's T-function

    :math:`T(h,a)=(2\pi)^{-1}\int_{0}^{a} \\frac{\exp(-\\frac{1}{2} h^2 (1+x^2))}{1+x^2} dx , (-\infty < h, a < +\infty).`

    Uses scipy.integrate.quad to evaluate the integral. 

    :type h: numpy.array
    :type a: numpy.array
    :returns: function values (array of |h| X |a|)
    :rtype:   numpy.array
    
    """
    f = lambda x,hh: (2*pi)**-1 * exp(-.5*hh**2*(1+x**2))/(1+x**2)

    if type(h) == ndarray and len(h.shape) > 1:
        if min(h.shape) > 1:
            raise DimensionalityError("natter.Auxiliary.Numerics.owensT: h may only be (n,1), (1,n) or (n,) in size!")
        else:
            h = reshape(array(h),(max(h.shape),))
    if type(a) == ndarray and len(a.shape) > 1:
        if min(a.shape) > 1:
            raise DimensionalityError("natter.Auxiliary.Numerics.owensT: a may only be (n,1), (1,n) or (n,) in size!")
        else:
            a = reshape(array(a),(max(a.shape),))
    if type(h) == ndarray and type(a) == ndarray:
        n = max(h.shape)
        m = max(a.shape)
    elif type(h) == ndarray:
        n = max(h.shape)
        m = 1
        a = reshape(a,(1,))
    elif type(a) == ndarray:
        n = 1
        m = max(a.shape)
        h = reshape(h,(1,))
    else:
        n=1
        m=1
    ret = zeros((n,m))
    for i in xrange(n):
        for j in xrange(m):
            ret[i,j] = quad(f,0.0,a[j],args=(h[i]))[0]
    return ret
        
        

def logsumexp(a, axis=None):
    """
    Evaluates :math:`\log(\sum_i \exp(a_i) )` in a bit smarter manner.

    :param a: positions where logsumexp will be evaluated
    :type a:  numpy.array
    :returns: function values
    :rtype:   numpy.array
    """
    if axis is None:
        # Use the scipy.maxentropy version.
        return maxentutils.logsumexp(a)
    a = asarray(a)
    shp = list(a.shape)
    shp[axis] = 1
    a_max = a.max(axis=axis)
    s = log(exp(a - a_max.reshape(shp)).sum(axis=axis))
    lse  = a_max + s
    return lse

def inv_digamma(y,niter=5):
    """
    Inverse of the digamma function with an algorithm by Tom Minka.
    
    A different algorithm is provided by Paul Fackler:
    http://www.american.edu/academic.depts/cas/econ/gaussres/pdf/loggamma.src

    :param y: Values where the inverse digamma will be evaluated
    :type y:  numpy.array
    :returns: function values
    :rtype:   numpy.array 
    """
    # Newton iteration to solve digamma(x)-y = 0
    s = shape(y)
    y = y.flatten()
    
    x = exp(y)+.5
    i = where(y <= -2.22)
    x[i] = -1.0/(y[i] - digamma(1.0))

    for i in range(niter):
        x -= (digamma(x)-y)/trigamma(x);
    return reshape(x,s)


def digamma(x):
    """
    Computed the digamma function for a whole array of numbers.

    :param x: array where the digamma function is to be evaluated
    :type x: numpy.array
    :returns: array with function values
    :rtype: numpy.array
    """
    if type(x) == float64 or type(x) == types.FloatType:
        return special.polygamma(0,x)
    else:
        s = shape(x)
        return  reshape(array([special.polygamma(0,e) for e in x.flatten()]),s)

def trigamma(x):
    """
    Computed the trigamma function for a whole array of numbers.

    :param x: array where the trigamma function is to be evaluated
    :type x: numpy.array
    :returns: array with function values
    :rtype: numpy.array
    """
    if type(x) == float64  or type(x) == types.FloatType:
        return special.polygamma(1,x)
    else:
        s = shape(x)
        return  reshape(array([special.polygamma(1,e) for e in x.flatten()]),s)


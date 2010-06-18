from scipy.maxentropy import maxentutils 
from numpy import asarray, log, exp, array, where, float64, shape, reshape, size
from scipy import special
import types
from Errors import DimensionalityError



def logsumexp(a, axis=None):
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
    INV_DIGAMMA    Inverse of the digamma function.
    
    inv_digamma(y) returns x such that digamma(x) = y.
    
    a different algorithm is provided by Paul Fackler:
    http://www.american.edu/academic.depts/cas/econ/gaussres/pdf/loggamma.src
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
    if type(x) == float64 or type(x) == types.FloatType:
        return special.polygamma(0,x)
    else:
        s = shape(x)
        return  reshape(array([special.polygamma(0,e) for e in x.flatten()]),s)

def trigamma(x):
    if type(x) == float64  or type(x) == types.FloatType:
        return special.polygamma(1,x)
    else:
        s = shape(x)
        return  reshape(array([special.polygamma(1,e) for e in x.flatten()]),s)


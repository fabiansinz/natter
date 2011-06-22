from __future__ import division
from scipy.maxentropy import maxentutils 
from numpy import asarray, log, exp, array, where, float64, shape, reshape,  pi, min,max, ndarray, zeros, atleast_1d, hstack, arange, remainder, isreal, all, conj, atleast_2d, zeros_like
from numpy.fft import fft, ifft
from scipy import special
from scipy.special import  gammaincc
from scipy.special import gamma as gammafunc, digamma
import types
from Errors import DimensionalityError
from scipy.integrate import quad
from mpmath import meijerg 



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


# def digamma(x):
#     """
#     Computed the digamma function for a whole array of numbers.

#     :param x: array where the digamma function is to be evaluated
#     :type x: numpy.array
#     :returns: array with function values
#     :rtype: numpy.array
#     """
#     if type(x) == float64 or type(x) == types.FloatType:
#         return special.polygamma(0,x)
#     else:
#         s = shape(x)
#         return  reshape(array([special.polygamma(0,e) for e in x.flatten()]),s)

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


            
    
def totalDerivativeOfIncGamma(x,a,b,da,db):
    """
    Computes the total derivative for the (non-normalized) incomplete gamma function, i.e.

    d/dx gamma(f(x))*gammainc(f(x),g(x))

    :param y: Positions where the function is to be computed.
    :param f: function handle for f
    :param g: function handle for g
    :param df: function handle for df/dx
    :param dg: function handle for dg/dx
    :returns: derivative values
    :rtype:   numpy.array
    
    """
    return digamma(a(x))*gammafunc(a(x))*da(x) + exp(-b(x))*b(x)**(a(x)-1)*db(x) \
     - (meijerg([[],[1,1],],[[0,0,a(x)],[]],b(x)) + log(b(x))*gammaincc(a(x),b(x))*gammafunc(a(x)))  * da(x)


# from scipy.org
def dct(x,n=None):
    """
    Discrete Cosine Transform

                      N-1
           y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                      n=0

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    Reference
    ---------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
    """
    x = atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    if x.shape[-1]<n:
        n_shape = x.shape[:-1] + (n-x.shape[-1],)
        xx = hstack((x,zeros(n_shape)))
    else:
        xx = x[...,:n]

    real_x = all(isreal(xx))
    if (real_x and (remainder(n,2) == 0)):
        xp = 2 * fft(hstack( (xx[...,::2], xx[...,::-2]) ))
    else:
        xp = fft(hstack((xx, xx[...,::-1])))
        xp = xp[...,:n]

    w = exp(-1j * arange(n) * pi/(2*n))

    y = xp*w

    if real_x:
        return y.real
    else:
        return y

def idct(x,n=None):
    """
    Inverse Discrete Cosine Transform

                       N-1
           x[k] = 1/N sum w[n]*y[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                       n=0

           w(0) = 1/2
           w(n) = 1 for n>0

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    Reference
    ---------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
    """

    x = atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    w = exp(1j * arange(n) * pi/(2*n))

    if x.shape[-1]<n:
        n_shape = x.shape[:-1] + (n-x.shape[-1],)
        xx = hstack((x,zeros(n_shape)))*w
    else:
        xx = x[...,:n]*w

    real_x = all(isreal(x))
    if (real_x and (remainder(n,2) == 0)):
        xx[...,0] = xx[...,0]*0.5
        yp = ifft(xx)
        y  = zeros(xx.shape,dtype=complex)
        y[...,::2] = yp[...,:n/2]
        y[...,::-2] = yp[...,n/2::]
    else:
        yp = ifft(hstack((xx, atleast_2d(zeros_like(xx[...,0])).transpose(), conj(xx[...,:0:-1]))))
        y = yp[...,:n]

    if real_x:
        return y.real
    else:
        return y

def idct2(x):
    a = idct(idct(x).transpose()).transpose()
    return a

def dct2(x):
    a = dct(dct(x).transpose()).transpose()
    return a

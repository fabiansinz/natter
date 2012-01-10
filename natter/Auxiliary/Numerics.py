from __future__ import division
from scipy.maxentropy import maxentutils 
from numpy import asarray, log, exp, array, where, float64, shape, reshape,  pi, min,max, ndarray, zeros, atleast_1d, hstack, arange, remainder, isreal, all, conj, atleast_2d, zeros_like, any, abs,mean
from numpy.fft import fft, ifft
from scipy import special
from scipy.special import  gammaincc
from scipy.special import gamma as gammafunc, digamma
import types
from Errors import DimensionalityError
from scipy.integrate import quad
from mpmath import meijerg 
from sys import stdout

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
    Computes the Discrete Cosine Transform
    :math:`y[k] = 2*\sum^{N-1}_{n=0}x[n]*\cos(\pi*k*(2n+1)/(2*N))`, 0 <= k < N.
    taken from scipy.org, for details see
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/

    :param x: 1D array 
    :type x: numpy.ndarray
    :param n: length of array, default None
    :type n: int
    :returns: 1D array
    :rtype: numpy.array
    
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
    Computes the Inverse Discrete Cosine Transform
    :math:`x[k] = 1/N \sum^{N-1}_{n=0} w[n]*y[n]*\cos(\pi*k*(2n+1)/(2*N))`, 0 <= k < N.
    :math:`w(0) = 1/2`
    :math:`w(n) = 1` for n>0
    Taken from scipy.org. For details see
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/

    :param x: 1D array 
    :type x: numpy.ndarray
    :param n: length of array, default None
    :type n: int
    :returns: 1D array
    :rtype: numpy.array
    
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
    """
    Computes 2D inverse DCT using idct() method on both directions

    :param x: 2D image
    :type x: numpy.ndarray
    :returns: 2D image
    :rtype: numpy.ndarray

    """
    a = idct(idct(x).transpose()).transpose()
    return a

def dct2(x):
    """
    Computes 2D DCT using dct() method on both directions

    :param x: 2D image
    :type x: numpy.ndarray
    :returns: 2D image
    :rtype: numpy.ndarray

    """   
    a = dct(dct(x).transpose()).transpose()
    return a

def invertMonotonicIncreasingFunction(f,y,xl,xu,tol=1e-6,maxiter = 10000):
    """
    Inverts a monotonically increasing function.

    :param f: function to be inverted
    :param y: desired output values
    :param xl: lower bounds for input x
    :type xl: numpy.ndarray
    :param xu: upper bounds for input x
    :type xu: numpy.ndarray
    :param tol: convergence tolerance
    :param maxiter: maximal number of iterations
    
    """
    yu = f(xu) - y
    while any(yu < 0.0):
        ind = where(yu < 0.0)
        xu[ind] = abs(xu[ind])*2.0

        yu[ind] = f(xu[ind]) - y[ind]

    yl = f(xl) - y
    while any(yl > 0.0):
        ind = where(yl > 0.0)
        xl[ind] = -abs(xl[ind])*2.0
        yl[ind] = f(xl[ind]) - y[ind]

    count = 0
    while any(xu-xl > tol) and count < maxiter:
        count += 1
        xm = 0.5*(xl + xu)
        ym = f(xm) - y

        ind = where(ym <= 0)
        xl[ind] = xm[ind]

        ind = where(ym > 0)
        xu[ind] = xm[ind]
        ym = f(0.5*(xl+xu)) - y
    #     stdout.flush()
    # stdout.write('\n')
    # stdout.flush()
    if count == maxiter:
        stdout.write('\tWarning in invertMonotonicIncreasingFunction: Inversion might not have been converged.\n')
        stdout.write('\tmax |xu-xl|=%.3g\t\tmax |ym|=%.3g\n' % (max(abs(xu-xl)),max(abs(ym))))
        stdout.flush()
        
    
    return 0.5*(xu+xl)
        
    
    

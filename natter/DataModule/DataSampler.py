from __future__ import division
from numpy import shape,  zeros, pi, NaN, isinf, isnan, any, array, reshape, dot,eye, ceil, arange, meshgrid,floor, sin, cos, vstack,sum,sqrt, arctan2
from numpy.random import rand, randn
from natter.DataModule import Data
from numpy.linalg import cholesky
from os import listdir
from sys import stdout

def gratings(p,omega0,phi0,deltaAlpha,deltaOmega,mu):
    """
    Creates a data object that contains gratings on pxp patches. The
    functional form of the grating is
    
    cos( 2*pi/p <w0,n-mu> + phi0 )

    :param p: patch size (patch is pxp)
    :type p: int or float
    :param omega0: frequency vector
    :type omega0: numpy.array of length 2
    :param phi0: initial phase
    :type phi0: float
    :param deltaAlpha: additive changes in the orientation of omega0
    :type deltaAlpha: numpy.array
    :param deltaOmega: additive changes in the length of omega0
    :type deltaOmega: numpy.array
    :param mu: center of the grating
    :type mu: numpy.array of length 2
    :returns: an array containing the orientations in the first and the frequencies in the second row, as well as a data object containing the gratings
    :rtype: numpy.array, natter.DataModule.Data
    """
    
    # rotates the frequency vector
    rot = lambda w,alpha: array([w[0]*cos(alpha) -w[1]*sin(alpha),w[0]*sin(alpha)+ w[1]*cos(alpha)])

    # sampling points
    [x,y] = meshgrid(arange(p),arange(p))
    X = vstack((x.reshape((1,p**2),order='F'),y.reshape((1,p**2),order='F')))

    # matrix that holds the gratings later
    G = zeros((p**2,len(deltaAlpha)*len(deltaOmega)))

    # compute spatial frequency corresponding to w and make deltaOmega a multiplicative factor
    f = sqrt(sum(omega0**2))
    deltaOmega /= f
    deltaOmega += 1.0

    # will store the actual orientations and frequencies later
    labels = zeros((2,len(deltaAlpha)*len(deltaOmega)))

    # generate gratings
    k = 0
    for alpha in deltaAlpha:
        for omega in deltaOmega:
            G[:,k] = cos(2*pi* dot(rot(omega0*omega/p,alpha),X) - 2*pi* dot(rot(omega0*omega/p,alpha),mu) + phi0)
            labels[0,k] = arctan2(omega0[1],omega0[0])+alpha
            labels[1,k] = f*omega
            k += 1
    return labels,Data(G)


def gauss(n,m,mu = None, sigma = None):
    """

    Samples m n-dimensional samples from a Gaussian with mean mu and covariance sigma.


    :param n: dimensionality
    :type n: int
    :param m: number of samples
    :type m: int
    :param mu: mean (default = zeros((n,1)))
    :type mu: numpy.array
    :param sigma: covariance matrix (default = eye(n))
    :type sigma: numpy.array
    :returns: Data object with sampled patches
    :rtype: natter.DataModule.Data
    
    """
    if not mu == None:
        mu = reshape(mu,(n,1))
    else:
        mu = zeros((n,1))
    if sigma == None:
        sigma = eye(n)
    return Data(dot(cholesky(sigma),randn(n,m))+mu,'Multivariate Gaussian data.')


def img2PatchRand(img, p, N):
    """

    Samples N pxp patches from img.

    The images are vectorized in FORTRAN/MATLAB style.

    :param img: Image to sample from
    :type img: numpy.array
    :param p: patch size
    :type p: int
    :param N: number of patches to sampleFromImagesInDir
    :type N: int
    :returns: Data object with sampled patches
    :rtype: natter.DataModule.Data
    
    """

    ny,nx = shape(img)

    x = floor( rand( N, 1) * ( nx - p + 1)) 
    y = floor( rand( N, 1) * ( ny - p + 1))

    p1 = p - 1
  
    X = zeros( ( p*p, N))

    for ii in xrange(int(N)):
        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0): 
            xi = floor( rand() * ( nx - p))
            yi = floor( rand() * ( ny - p))
            ptch = img[ yi:yi+p1+1, xi:xi+p1+1]
            X[:,ii] = ptch.flatten('F')
  
    name = "%d %dX%d patches" % (N,p,p)
    return Data(X, name)

def sampleFromImagesInDir(dir, m, p, loadfunc, samplefunc=img2PatchRand):
    """

    Samples m patches from images in dir by loading them with
    loadfunc(filename) and sampling patches via samplefunc(img, p,
    ceil(m/#images)).

    :param dir: Directory containing the images
    :type dir: string
    :param m: number of images to samplefunc
    :type m: int
    :param p: patchsize
    :type p: int
    :param loadfun: function handle of the load function
    :param samplefunc: function handle of the sampling function
    :returns: Data object with the sampled image patches
    :rtype: natter.DataModule.Data

    
    """
    files = listdir(dir)
    M = len(files)
    mpf = ceil(m/M)

    # load and sample first image
    dat = img2PatchRand(loadfunc(dir + files[0]), p, mpf)
    
    for i in xrange(1,M):
        print "Loading %d %dx%d patches from %s" %(mpf,p,p,dir + files[i] )
        stdout.flush()
        dat.append(img2PatchRand(loadfunc(dir + files[i]), p, mpf))
    return dat
        
        

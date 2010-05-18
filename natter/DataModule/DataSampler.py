from numpy import shape, floor, zeros, NaN, isinf, isnan, any, array, reshape, dot,eye
from numpy.random import rand, randn
from Data import Data
from numpy.linalg import cholesky


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

    for ii in xrange(N):
        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0): 
            xi = floor( rand() * ( nx - p))
            yi = floor( rand() * ( ny - p))
            ptch = img[ yi:yi+p1+1, xi:xi+p1+1]
            X[:,ii] = ptch.flatten('F')
  
    name = "%d %dX%d patches" % (N,p,p)
    return Data(X, name)

def sampleFromImagesInDir(dir, m, p, loadfunc, samplefunc):
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
    
    pass

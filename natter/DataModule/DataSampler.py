from __future__ import division
from numpy import shape,  zeros, pi, NaN, isinf, isnan, any, array, reshape, dot,eye, ceil, meshgrid,floor, cos, vstack,sum,   ones, kron, arange,max, real, imag,round
from numpy.random import rand, randn,poisson
from natter.DataModule import Data
from numpy.linalg import cholesky
from os import listdir
from sys import stdout
from numpy.fft import fft2

#from matplotlib.pyplot import *

def gratings(p,T,omega,tau):
    """
    Creates a data object that contains gratings on pxp patches. The
    maximal and minimal amplitude of the gratings is 1. The functional
    form of the grating is
    
    :math:`\\cos( \\frac{2*\\pi}{p} \\langle\\mathbf \\omega,\\mathbf n\\rangle + \\frac{2*\\pi}{T} \\tau\\cdot t )`

    where :math:`t\in \{0,...,T-1\}` and :math:`\mathbf\omega\in\{0,...p\}^2`.
    T is the length of the vector tau.

    The total number of patches which is created is m*T*|tau|, where m
    is the number of columns of omega and T is the number of time
    points and |tau| the length of tau. The ordering in the data
    object will be the following

    
    +---------------------------------------+---------------------------------+-----+
    |            tau[0]                     |            tau[1]               | ... |
    +----------------+----------------+-----+----------------+----------------+-----+
    |    omega[0]    |    omega[1]    | ... |    omega[0]    |    omega[1]    | ... |
    +----------------+----------------+-----+----------------+----------------+-----+
    | 0, 1, ..., T-1 | 0, 1, ..., T-1 | ... | 0, 1, ..., T-1 | 0, 1, ..., T-1 | ... |
    +----------------+----------------+-----+----------------+----------------+-----+
    
    i.e. the first T patches have spatial frequency omega[:,0] and
    temporal frequency tau[0], the next T patches have spatial
    frequency omega[:,1] and spatial frequency tau[0], and so on.

    :param p: patch size (patch is pxp)
    :type p: int 
    :param T: patch size (patch is pxp)
    :type T: int 
    :param omega: frequency vectors
    :type omega: numpy.array of shape 2 x m
    :param tau: temporal frequencies
    :type tau: numpy.array of shape 1 x |tau|
    :returns: a data object containing the gratings 
    :rtype: natter.DataModule.Data
    """

    
    if len(omega.shape) == 1:
        omega = omega.reshape((2,1))
    
    m = omega.shape[1]
    t = max(tau.shape)

    
    # sampling points
    [x,y] = meshgrid(arange(p),arange(p))
    X = vstack((x.reshape((1,p**2),order='F'),y.reshape((1,p**2),order='F'))) # X = 2 X p^2
    X = kron(ones((1,T*m*t)),X) # make T*m*t copies of X
    X = vstack((X,kron(ones((1,m*t)),kron(arange(T),ones((1,p**2))))))
    
    # frequencies
    W = kron(omega / p,ones((1,T*p**2))) 
    W = kron(ones((1,t)),W)
    W = vstack((W,kron(tau/T,ones((1,T*m*p**2)))))

    g = cos(2*pi*sum(W*X,0))
    G = zeros((p**2,m*T))

    pointer = 0
    index = 0
    for i in xrange(T):
        for j in xrange(m):
            G[:,index] = g[pointer:pointer+p**2]
            index+=1
            pointer += p**2


    return Data(G,"Gratings: %i spatial and %i temporal frequencies" % (m,T))


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

    p1 = p - 1
  
    X = zeros( ( p*p, N))

    stdout.flush()
    for ii in xrange(int(N)):
        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0): 
            xi = floor( rand() * ( nx - p))
            yi = floor( rand() * ( ny - p))
            ptch = img[ yi:yi+p1+1, xi:xi+p1+1]
            X[:,ii] = ptch.flatten('F')
  
    name = "%d %dX%d patches" % (N,p,p)
    return Data(X, name)


def eyeMovementGenerator(dir,loadfunc, p,tau,sigma,randStayTime = True):
    """
    Simulates eye movements by sampling patches with the following procedure:

    1) an image is loaded with loadfunc from the directory dir
    2) the number N of pxp patches that will be sampled from that image is sampled from a Poisson distribution with mean tau. If randStayTime==False then tau is used instead.
    3) a random starting location (uniformly distributed) is sampled
    4) a patch is sampled form that location
    5) a new location is sampled from Brownian motion with std sigma, if the border of the image is reached, the sample is rejected and resampled
    6) if N is reached, goto 1), otherwise goto 4)

    :param dir: directory containing only images
    :type dir: string
    :param loadfun: function handle of the image load function which takes a string and returns a numpy array (the image)
    :param p: patchsize
    :type p: int
    :param tau: mean of Poisson distribution
    :type tau: float
    :param sigma: std of the 2D Brownian motion
    :type sigma: float
    :param N: if not None, then N patches will be sampled in each step
    :type N: int
    :returns:  Data object with sampled patches
    :rtype: natter.DataModule.Data
    """
    if dir[-1] != '/':
        dir += '/'
        
    files = listdir(dir)
    M = len(files)
    sampleImg = True
    t = 0
    I = None
    N = tau
    while True:
        if sampleImg:
            # if I is not None:
            #     show()
            #     raw_input()
            t = 0
            if randStayTime:
                N = poisson(tau)
            filename = dir + files[int(floor(rand()*M))]
            I = loadfunc(filename)
            
            ny,nx = I.shape
            # imshow(I,interpolation='nearest',cmap=cm.gray)
            xi = floor( rand() * ( nx - p))
            yi = floor( rand() * ( ny - p))
            sampleImg = False
            stdout.write('\tSampling %i %i X %i patches from %s\n' % (N,p,p,filename))
            stdout.flush()

        
        ptch = I[ yi:yi+p, xi:xi+p]
        #plot(xi+p/2.0,yi+p/2,'ro')
        tmp = round(randn(2)*sigma)
        while yi + tmp[0]+p >= ny or xi + tmp[1] +p >= nx or  yi + tmp[0] < 0 or xi + tmp[1] < 0:
            tmp = round(randn(2)*sigma)
        yi += tmp[0]
        xi += tmp[1]
        # stdout.write("(%i,%i)" % (yi,xi))
        # stdout.flush()
        t += 1
        ptch = ptch.flatten()
        if any(isnan(ptch)) or any(isinf(ptch)):
            t -= 1
        else:
            yield ptch.flatten('F')

        if t == N:
            sampleImg = True
    
def directoryIterator(dir,m,p,loadfunc,samplefunc=img2PatchRand):
    """

    Iterator to sample ceil(m/#images) patches from each image in dir
    by loading them with loadfunc(filename) and sampling patches via
    samplefunc(img, p, ceil(m/#images)). The iterator yields a image
    patch per call.

    :param dir: Directory containing the images
    :type dir: string
    :param m: number of images to samplefunc
    :type m: int
    :param p: patchsize
    :type p: int
    :param loadfun: function handle of the load function
    :param samplefunc: function handle of the sampling function
    
    """
    files = listdir(dir)
    M = len(files)
    mpf = ceil(m/M)

    # load and sample first image
    
    for i in xrange(M):
        I = loadfunc(dir + files[i])
        print "\tLoading %d %dx%d patches from %i X %i size %s" %(mpf,p,p,I.shape[0],I.shape[1],dir + files[i] )
        stdout.flush()
        X = samplefunc(I, p, mpf).X
        for j in xrange(X.shape[1]):
            yield X[:,j]
    return 

def sampleWithIterator(theIterator,m,transformfunc = None):
    """
    Uses the iterator to sample m patches from it. theIterator must
    return a data point at a time.

    :param theIterator: Iterator that returns data poitns
    :type theIterator: iterator
    :param m: number of patches to sample
    :type m: int
    :param transformfunc: function that get applied to a patch once it is sampled.
    """
    count = 1
    x0 = theIterator.next()
    n = max(x0.shape)
    X = zeros((n,m))
    if transformfunc is not None:
        X[:,0] = transformfunc(x0)
    else:
        X[:,0] = x0

    for x in theIterator:

        if transformfunc is not None:
            X[:,count] = transformfunc(x)
        else:
            X[:,count] = x
            
        count += 1
        if count == m:
            break
    if transformfunc is not None:
        name = '%i transformed data points sampled with iterator.' % (m, )
    else:
        name = '%i data points sampled with iterator.' % (m, )
        
    return Data(X,name)
    
def sample(theIterator,m,transformfunc = None):
    """
    See doc for sampleWithIterator.
    """    
    return sampleWithIterator(theIterator,m,transformfunc)

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
    print "Loading %d %dx%d patches from %s" %(mpf,p,p,dir + files[0] )
    stdout.flush()
    dat = img2PatchRand(loadfunc(dir + files[0]), p, mpf)
    
    for i in xrange(1,M):
        print "Loading %d %dx%d patches from %s" %(mpf,p,p,dir + files[i] )
        stdout.flush()
        dat.append(samplefunc(loadfunc(dir + files[i]), p, mpf))
    return dat
        
        
# if __name__=="__main__":
#     from numpy import *
#     from natter.DataModule import DataSampler
#     from matplotlib.pyplot import show

#     n = 5
#     t = arange(0,2*pi,2*pi/100)
#     W = vstack((cos(t),sin(t)))
#     dat = DataSampler.gratings(n,9,array([0,1]),array([1]))

#     z = fft2(reshape(dat.X[:,1],(n,n),order='F'))
#     for i in xrange(n):
#         print "\t".join(["%.2f + i%.2f" % (real(elem), imag(elem)) for elem in z[i,:]])
        

    

    

    

from __future__ import division
from numpy import shape,  zeros, pi, NaN, isinf, isnan, any, array, reshape, dot,eye, ceil, meshgrid,floor, cos, vstack,sum, ones, kron, arange,max, empty_like, hstack, fft, sqrt, exp, round, asarray
from numpy.random import rand, randn, randint, poisson
from natter.DataModule import Data
#from natter.Distributions import Uniform
from natter.Auxiliary.ImageUtils import shiftImage
from numpy.linalg import cholesky
from os import listdir
from sys import stdout
import Image
#from numpy.fft import fft2

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
    


def sampleWithIterator(theIterator,m):
    """
    Uses the iterator to sample m patches from it. theIterator must
    return a data point at a time.

    :param theIterator: Iterator that returns data poitns
    :type theIterator: iterator
    :param m: number of patches to sample
    :type m: int
    """
    count = 1
    x0 = theIterator.next()
    n = max(x0.shape)
    X = zeros((n,m))
    X[:,0] = x0
    for x in theIterator:
        X[:,count] = x
        count += 1
        if count == m:
            break
    return Data(X,'%i data points sampled with iterator.' % (m, ))
    

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
        

def randPatchWithBorderIterator(dir, p, samples_per_file, loadfunc, borderwidth=0, orientation='F'):
    """

    Samples pxp patches from img.

    The images are vectorized in FORTRAN/MATLAB style by default.

    :param dir: Directory to sample images from
    :type dir: string
    :param p: patch size
    :type p: int
    :param samples_per_file: number of patches to sample from one image
    :type samples_per_file: int   
    :param loadfunc: function handle of the load function
    :param borderwidth: width of the border of the source image which cannot be used for sampling. Default 0.
    :type borderwidth: int
    :param orientation: 'C' (C/Python, row-major) or 'F' (FORTRAN/MATLAB, column-major) vectorized patches
    :type orientation: string
    :returns: Iterator that samples from all files
    :rtype: Iterator
    
    """
    if dir[-1] != '/':
        dir += '/'
        
    files = listdir(dir)
    number_files = len(files)
    sampleImg = True

    while True:
        if sampleImg:
            sample_index = 0
            filename = dir + files[int(rand()*number_files)]
            img = loadfunc(filename)
            
            ny,nx = img.shape
            width_limit = nx - p - borderwidth
            height_limit = ny - p - borderwidth
            sampleImg = False
            stdout.write('\tSampling %i X %i patches from %s\n' % (p,p,filename))
            stdout.flush()
           
        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0): 
            xi = randint(low=borderwidth, high=width_limit)
            yi = randint(low=borderwidth, high=height_limit)
            ptch = img[ yi:yi+p, xi:xi+p]
            X = ptch.flatten(orientation)
        
        sample_index += 1
        yield X
        if sample_index == samples_per_file:
            sampleImg = True
  
    return

def randShiftSequenceWithBorderIterator(dir, p, samples_per_file, loadfunc, borderwidth=0, orientation='F', shiftDistribution=None):
    """

    Samples pxp sequences from images in dir

    The images are vectorized in FORTRAN/MATLAB style by default.

    :param dir: Directory to sample images from
    :type dir: string
    :param p: patch size
    :type p: int
    :param samples_per_file: number of patches to sample from one image. All samples have same shift.
    :type samples_per_file: int    
    :param loadfunc: function handle of the load function
    :param borderwidth: width of the border of the source image which cannot be used for sampling. Default 0.
    :type borderwidth: int
    :param orientation: 'C' (C/Python, row-major) or 'F' (FORTRAN/MATLAB, column-major) vectorized patches
    :type orientation: string
    :param shiftDistribution: 2D distribution to sample shift steps from
    :type shiftDistribution: natter.Distributions.Distribution
    :returns: Iterator that samples from all files
    :rtype: Iterator
    
    """
    if shiftDistribution == None:
        raise ValueError, 'shiftDistribution cannot be None. Use e.g. Uniform(n=2, low=-1.0, high=1.0)'

    if dir[-1] != '/':
        dir += '/'
        
    files = listdir(dir)
    number_files = len(files)
    sampleImg = True

    while True:
        if sampleImg:
            sample_index = 0
            filename = dir + files[int(rand()*number_files)]
            img = loadfunc(filename)
            img2 = empty_like(img)
            shift = shiftDistribution.sample(1).X.flatten()
            img2[borderwidth:-borderwidth, borderwidth:-borderwidth] = shiftImage(img[borderwidth:-borderwidth, borderwidth:-borderwidth], shift)
            
            ny,nx = img.shape
            offset = ceil((shift + abs(shift))/2)
            width_limit = nx - p - borderwidth - abs(ceil(shift[-1]))
            height_limit = ny - p - borderwidth - abs(ceil(shift[0]))
            sampleImg = False
            stdout.write('\tSampling %i X %i patches from %s with shift ('%(p,p,filename) \
                         + shift.size*'%.3f, '%tuple(shift) + '\b\b)\n')
            stdout.flush()
           
        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0): 
            xi = randint(low=borderwidth+offset[-1], high=width_limit)
            yi = randint(low=borderwidth+offset[0], high=height_limit)
            X = img[ yi:yi+p, xi:xi+p].flatten(orientation)
            Y = img2[ yi:yi+p, xi:xi+p].flatten(orientation)
            ptch = hstack((X.flatten(), Y.flatten()))
        
        sample_index += 1
        yield X, Y
        if sample_index == samples_per_file:
            sampleImg = True
  
    return

def circulantPinkNoiseIterator(p, powerspectrum_sample_size, patchSampler, orientation='F'):
    """

    Samples pxp patches from circulant pink noise with power spectrum from patchSampler.

    The images are vectorized in FORTRAN/MATLAB style by default.

    :param dir: Directory to sample images from
    :type dir: string
    :param p: patch size
    :type p: int
    :param powerspectrum_sample_size: number of patches to sample from sampler for power spectrum
    :type powerspectrum_sample_size: int   
    :param orientation: 'C' (C/Python, row-major) or 'F' (FORTRAN/MATLAB, column-major) vectorized patches
    :type orientation: string
    :returns: Iterator that samples from all files
    :rtype: Iterator
    
    """

    patches = zeros((p**2, powerspectrum_sample_size))
    stdout.write('Initialize power spectrum of circulant pink noise generator. This may take a moment...\n')
    stdout.flush()
    for ii in xrange(powerspectrum_sample_size):
        patches[:,ii] = patchSampler.next()
    PATCHES = fft.fft2(patches.reshape(p,p,powerspectrum_sample_size), axes=(0,1))
    powerspec = (PATCHES*PATCHES.conj()).mean(2).real
    stdout.write('Powerspectrum completed.\n')

    while True:
        phi = rand(p,p)*2*pi - pi
        X = fft.ifft2(sqrt(powerspec)*exp(1J*phi), axes=(0,1)).real.flatten(orientation)
        yield X
  
    return

def sampleSequenceWithIterator(theIterator, m):
    """
    Uses the iterator to sample a sequence of m pairs of patches from it.
    theIterator must return a pair of data points at a time. Pairs are
    stored at column i and i+m.

    :param theIterator: Iterator that returns data poitns
    :type theIterator: iterator
    :param m: number of patch pairs to sample
    :type m: int
    :returns: Data object with 2*m samples
    :rtype: natter.DataModule.Data    
    """
    count = 1
    x0,y0 = theIterator.next()
    n = x0.size
    X = zeros((n,2*m))
    X[:,0] = x0
    X[:,m] = y0
    for sample in theIterator:
        X[:,count] = sample[0]
        X[:,count+m] = sample[1]
        count += 1
        if count == m:
            break
    return Data(X,'Sequence of %i data pairs sampled with iterator.' % (m, ))


def randRotationSequenceWithBorderIterator(dir, p, samples_per_file, loadfunc, borderwidth=0, orientation='F', rotationDistribution=None):
    """

    Samples pxp sequences from images in dir. 

    The images are vectorized in FORTRAN/MATLAB style by default.

    :param dir: Directory to sample images from
    :type dir: string
    :param p: patch size
    :type p: int
    :param samples_per_file: number of patches to sample from one image. All samples have same shift.
    :type samples_per_file: int    
    :param loadfunc: function handle of the load function
    :param borderwidth: width of the border of the source image which cannot be used for sampling. Default 0.
    :type borderwidth: int
    :param orientation: 'C' (C/Python, row-major) or 'F' (FORTRAN/MATLAB, column-major) vectorized patches
    :type orientation: string
    :param rotationDistribution: 1D distribution to sample shift steps from
    :type rotationDistribution: natter.Distributions.Distribution
    :returns: Iterator that samples from all files
    :rtype: Iterator
    
    """
    if rotationDistribution == None:
        raise ValueError, 'rotationDistribution cannot be None. Use e.g. Uniform(n=1, low=0.0, high=360.0)'

    if dir[-1] != '/':
        dir += '/'
        
    files = listdir(dir)
    number_files = len(files)
    sampleImg = True
    # to avoid artifacts from the undefined regions outside of the patch during rotation
    # we extend the source patch such that there are no undefined pixels 
    width_ext = ceil(p * (sqrt(2)-1)) 
    w = p+2*width_ext

    while True:
        if sampleImg:
            sample_index = 0
            filename = dir + files[int(rand()*number_files)]
            img = loadfunc(filename)[borderwidth:-borderwidth, borderwidth:-borderwidth]
            
            ny,nx = img.shape
            width_limit = nx - w
            height_limit = ny - w
            sampleImg = False
            stdout.write('\tSampling %i X %i rotating patch sequences from %s\n'%(p,p,filename))
            stdout.flush()
           
        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0): 
            xi = randint(low=0, high=width_limit)
            yi = randint(low=0, high=height_limit)
            shift = rand()*360
            Xsource = img[ yi:yi+w, xi:xi+w]
            Ysource = asarray(Image.fromarray(Xsource).rotate(shift))
            
            X = Xsource[width_ext:width_ext+p,width_ext:width_ext+p].flatten(orientation)
            Y = Ysource[width_ext:width_ext+p,width_ext:width_ext+p].flatten(orientation)
            ptch = hstack((X.flatten(), Y.flatten()))
        
        sample_index += 1
        yield X, Y
        if sample_index == samples_per_file:
            sampleImg = True
  
    return

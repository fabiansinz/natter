from __future__ import division
from numpy import shape,  zeros, pi, NaN, isinf, isnan, any, array, reshape, dot, eye, ceil, meshgrid,floor, cos, vstack, sum, ones, kron, arange,max, empty_like, hstack, fft, sqrt, exp, round, asarray, linspace, sin
from numpy.random import rand, randn, randint, poisson, exponential
from natter.DataModule import Data
from natter.Auxiliary.ImageUtils import shiftImage, bilinearInterpolation
from numpy.linalg import cholesky
from os import listdir
from sys import stdout
from scipy.ndimage.interpolation import rotate, zoom
#from numpy.fft import fft2



def gratings(p,T,omega,tau):
    """
    Creates a data object that contains gratings on pxp patches. The
    maximal and minimal amplitude of the gratings is 1. The functional
    form of the grating is

    :math:`\\cos( \\frac{2*\\pi}{p} \\langle\\mathbf \\omega,\\mathbf n\\rangle + \\frac{2*\\pi}{T} \\tau\\cdot t )`

    where :math:`t\in \{0,...,T-1\}` and :math:`\mathbf\omega\in\{0,...p\}^2`.
    T is the length of the vector tau.

    The total number of patches which is created is m*T*len(tau), where m
    is the number of columns of omega and T is the number of time
    points and len(tau) the length of tau. The ordering in the data
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
    :type tau: numpy.array of shape 1 x len(tau)
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


def eyeMovementGenerator(dir,loadfunc, p,N,sigma, mu, sampFreq,plot=False):
    """

    - change image all N saccades
    - sigma is std of Gaussian for Brownian motion

    """
    if plot:
        # plotting for debugging
        from matplotlib.pyplot import show, figure,cm

    if dir[-1] != '/':
        dir += '/'

    files = listdir(dir)
    M = len(files)
    t = 0
    I = None



    while True:
        xiold = NaN
        yiold = NaN

        # plot for debugging
        if I is not None and plot:
            show()
            raw_input()

        filename = dir + files[randint(M)]
        I = loadfunc(filename)
        samplecounter = 0
        if plot:
            # plot for debugging
            fig = figure(figsize=(4.86,3.5),dpi=300)
            ax = fig.add_axes([.0,.0,1.,1.])
            ax.imshow(I,interpolation='nearest',cmap=cm.gray)
            ax.axis('tight')



        ny,nx = I.shape
        stdout.write('\tLoaded image %s' % (filename,))
        stdout.flush()


        m = 0 # saccade counter
        while m < N:
            t = 0
            stayTime = exponential(scale=mu)


            xi = randint(nx - p)
            yi = randint( ny - p)

            while t < sampFreq*stayTime:
                if plot:
                    ax.plot([xiold+p/2.0,xi+p/2.0],[yiold+p/2.0,yi+p/2.0],'o-r')

                ptch = I[ yi:yi+p, xi:xi+p]
                yield ptch.flatten('F')

                tmp = round(sqrt(1.0/sampFreq)*randn(2)*sigma)
                while yi + tmp[0]+p >= ny or xi + tmp[1] +p >= nx or  yi + tmp[0] < 0 or xi + tmp[1] < 0:
                    tmp = round(sqrt(1.0/sampFreq)*randn(2)*sigma)

                xiold = xi
                yiold = yi
                yi += tmp[0]
                xi += tmp[1]

                samplecounter += 1
                t += 1

            m+=1
        stdout.write('\tFetched %i patches \n'% (samplecounter,))
        stdout.flush()






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

    Samples pxp sequences from images in dir. Transformation applied if a rotation
    around the center of the image patch, rotation angle (in degrees) is sampled from
    rotationDistribution. Rotates a larger patch than the one retuned to avoid artifacts.

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
            phi = rotationDistribution.sample(1).X[0,0]
            Xsource = img[ yi:yi+w, xi:xi+w]
            Ysource = rotate( Xsource, phi, reshape=False)

            X = Xsource[width_ext:width_ext+p,width_ext:width_ext+p].flatten(orientation)
            Y = Ysource[width_ext:width_ext+p,width_ext:width_ext+p].flatten(orientation)
            ptch = hstack((X.flatten(), Y.flatten()))

        sample_index += 1
        yield X, Y
        if sample_index == samples_per_file:
            sampleImg = True

    return

def randScalingSequenceWithBorderIterator(dir, patch_size, samples_per_file, loadfunc, borderwidth=0, orientation='F', scalingDistribution=None, upper_limit=2.0, lower_limit=0.5):
    """

    Samples pxp sequences from images in dir. Transformation applied is scaling. If scalingDistribution is
    1D then isometric scaling is applied, otherwise first dimension is scaling in x direction (width) and
    second dimension along y direction (height).
    CAUTION: If scalingDistribution returns only samples which are smaller than -patch_size the sampling
    will end in an infinite loop.

    The images are vectorized in FORTRAN/MATLAB style by default.

    :param dir: Directory to sample images from
    :type dir: string
    :param patch_size: patch size
    :type patch_size: int
    :param samples_per_file: number of patches to sample from one image. All samples have same shift.
    :type samples_per_file: int
    :param loadfunc: function handle of the load function
    :param borderwidth: width of the border of the source image which cannot be used for sampling. Default 0.
    :type borderwidth: int
    :param orientation: 'C' (C/Python, row-major) or 'F' (FORTRAN/MATLAB, column-major) vectorized patches
    :type orientation: string
    :param scalingDistribution: 1D or 2D distribution to sample shift steps from
    :type scalingDistribution: natter.Distributions.Distribution
    :param upper_limit: Upper limit for scaling (default 2.0)
    :type upper_limit: Float
    :param lower_limit: Lower limit for scaling (default 0.5)
    :returns: Iterator that samples from all files
    :type lower_limit: Float
    :rtype: Iterator

    """
    if scalingDistribution == None:
        raise ValueError, 'scalingDistribution cannot be None. Use e.g. Uniform(n=2, low=0.5, high=2.0)'
    elif scalingDistribution.param['n'] > 2:
        stdout.write('WARNING: scalingDistribution has more dimensions than needed. Using only first 2.\n')


    if dir[-1] != '/':
        dir += '/'

    files = listdir(dir)
    number_files = len(files)
    sampleImg = True
    # to avoid artifacts from the undefined regions outside of the patch during rotation
    # we extend the source patch such that there are no undefined pixels

    while True:
        if sampleImg:
            sample_index = 0
            filename = dir + files[int(rand()*number_files)]
            img = loadfunc(filename)[borderwidth:-borderwidth, borderwidth:-borderwidth]

            ny,nx = img.shape
            sampleImg = False
            stdout.write('\tSampling %i X %i scaling patch sequences from %s\n'%(patch_size,patch_size,filename))
            stdout.flush()
            width_limit_up = floor(nx - upper_limit*patch_size)
            height_limit_up = floor(ny - upper_limit*patch_size)
            width_limit_low = ceil(upper_limit/2.0*patch_size)
            height_limit_low = ceil(upper_limit/2.0*patch_size)


        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0):
            randSample = scalingDistribution.sample(1).X
            if randSample.size == 1:
                scale = array((randSample[0,0], randSample[0,0]))
            else:
                scale = array((randSample[0,0], randSample[1,0]))

            if scale[0] > upper_limit:
                scale[0] = upper_limit
            elif scale[0] < lower_limit:
                scale[0] = lower_limit
            if scale[1] > upper_limit:
                scale[1] = upper_limit
            elif scale[1] < lower_limit:
                scale[1] = lower_limit

            new_width = round(patch_size*scale[0])
            new_height = round(patch_size*scale[1])

            xi = randint(low=width_limit_low, high=width_limit_up)
            yi = randint(low=height_limit_low, high=height_limit_up)

            yinew = int((yi + patch_size/2) - new_height/2)
            xinew = int((xi + patch_size/2) - new_width/2)

            #stdout.write('%i %i %i %i %i %i %i\n'%(sample_index, xi, yi, xinew, yinew, new_height, new_width))

            Xsource = img[ yinew:yinew+new_height, xinew:xinew+new_width]
            Ysource = zoom(Xsource, (patch_size/new_height, patch_size/new_width))

            X = img[ yi:yi+patch_size, xi:xi+patch_size].flatten(orientation)
            Y = Ysource.flatten(orientation)
            ptch = hstack((X.flatten(), Y.flatten()))

        sample_index += 1
        yield X, Y
        if sample_index == samples_per_file:
            sampleImg = True

    return

def slidingWindowWithBorderIterator(dir, patch_size, samples_per_file, loadfunc, borderwidth=0, orientation='F', translationDistribution=None, rotationDistribution=None, scalingDistribution=None, upper_limit=2.0, lower_limit=0.5, debug=False):
    """

    Samples pxp sequences from images in dir. Transformations applied are translation, rotation, and scaling
    according to the given distributions. If one distribution is None the corresponding transformation is
    not applied.
    If scalingDistribution is 1D then isometric scaling is applied, otherwise first dimension is scaling in
    x direction (width) and second dimension along y direction (height).
    CAUTION: If scalingDistribution returns only samples which are smaller than -patch_size the sampling
    will end in an infinite loop.

    The images are vectorized in FORTRAN/MATLAB style by default.

    :param dir: Directory to sample images from
    :type dir: string
    :param patch_size: patch size
    :type patch_size: int
    :param samples_per_file: number of patches to sample from one image. All samples have same shift.
    :type samples_per_file: int
    :param loadfunc: function handle of the load function
    :param borderwidth: width of the border of the source image which cannot be used for sampling. Default 0.
    :type borderwidth: int
    :param orientation: 'C' (C/Python, row-major) or 'F' (FORTRAN/MATLAB, column-major) vectorized patches
    :type orientation: string
    :param translationDistribution: 2D distribution to sample shift steps from
    :type translationDistribution: natter.Distributions.Distribution
    :param rotationDistribution: 1D distribution to sample shift steps from
    :type rotationDistribution: natter.Distributions.Distribution
    :param scalingDistribution: 1D or 2D distribution to sample shift steps from
    :type scalingDistribution: natter.Distributions.Distribution
    :param upper_limit: Upper limit for scaling (default 2.0)
    :type upper_limit: Float
    :param lower_limit: Lower limit for scaling (default 0.5)
    :returns: Iterator that samples from all files
    :type lower_limit: Float
    :rtype: Iterator

    """

    if translationDistribution == None:
        raise ValueError, 'translationDistribution cannot be None. Use e.g. Uniform(n=2, low=-5.0, high=5.0)'
    elif translationDistribution.param['n'] > 2:
        stdout.write('WARNING: translationDistribution has more dimensions than needed. Using only first 2.\n')
    if rotationDistribution == None:
        raise ValueError, 'rotationDistribution cannot be None. Use e.g. Uniform(n=1, low=0.0, high=2*numpy.pi)'
    elif rotationDistribution.param['n'] > 1:
        stdout.write('WARNING: rotationDistribution has more dimensions than needed. Using only first.\n')
    if scalingDistribution == None:
        raise ValueError, 'scalingDistribution cannot be None. Use e.g. Uniform(n=1, low=0.0, high=360.0)'
    elif scalingDistribution.param['n'] > 2:
        stdout.write('WARNING: scalingDistribution has more dimensions than needed. Using only first 2.\n')


    if dir[-1] != '/':
        dir += '/'

    files = listdir(dir)
    number_files = len(files)
    sampleImg = True
    seed = zeros(2)
    pw = patch_size
    ph = patch_size

    while True:
        if sampleImg:
            sample_index = 0
            filename = dir + files[int(rand()*number_files)]
            img = loadfunc(filename)[borderwidth:-borderwidth, borderwidth:-borderwidth]

            ny,nx = img.shape
            sampleImg = False
            stdout.write('\tSampling %i X %i patch sequences from %s\n'%(patch_size,patch_size,filename))
            stdout.flush()
            width_limit_up = floor(nx - upper_limit*patch_size)
            height_limit_up = floor(ny - upper_limit*patch_size)
            width_limit_low = ceil(upper_limit/2.0*patch_size)
            height_limit_low = ceil(upper_limit/2.0*patch_size)
            seed[1] = randint(low=height_limit_low, high=height_limit_up)
            seed[0] = randint(low=width_limit_low, high=width_limit_up)

            x_center = linspace(seed[0]-(pw-1)/2, seed[0]+(pw-1)/2, pw, True)
            y_center = linspace(seed[1]-(ph-1)/2, seed[1]+(ph-1)/2, ph, True)
            x_grid, y_grid = meshgrid(x_center, y_center)
            grid_vectors_abs = vstack((x_grid.flatten(), y_grid.flatten()))
            grid_vectors_rel = grid_vectors_abs - seed.reshape(2,1)
            patch = zeros((patch_size**2))
            prev_patch = zeros((patch_size**2))
            bilinearInterpolation(img, grid_vectors_abs, patch )
            Rall = eye(2)

        alpha = rotationDistribution.sample(1).X[0,0]
        randSample = scalingDistribution.sample(1).X
        if randSample.size == 1:
            scale = array((randSample[0,0], randSample[0,0]))
        else:
            scale = array((randSample[0,0], randSample[1,0]))

        if scale[0] > upper_limit:
            scale[0] = upper_limit
        elif scale[0] < lower_limit:
            scale[0] = lower_limit
        if scale[1] > upper_limit:
            scale[1] = upper_limit
        elif scale[1] < lower_limit:
            scale[1] = lower_limit

        randSample = translationDistribution.sample(1).X
        if randSample.size == 1:
            shift = array((randSample[0,0], randSample[0,0]))
        else:
            shift = array((randSample[0,0], randSample[1,0]))

        R = array(((cos(alpha), sin(alpha)),(-sin(alpha),cos(alpha))))
        S = array(((scale[0], 0),(0, scale[1])))

        seed += shift
        grid_vectors_rel = dot(R, dot(Rall, dot(S, dot( Rall.T, grid_vectors_rel))))
        Rall = dot(R, Rall)
        grid_vectors_abs = grid_vectors_rel + seed.reshape(2,1)
        #print 'seed:', seed
        #print 'alpha', alpha, 'scale', scale, 'shift', shift

        if (grid_vectors_abs < 0).any() \
           or (grid_vectors_abs[0,:] > nx-1).any() \
           or (grid_vectors_abs[1,:] > ny-1).any():
            sampleImg = True
            print 'Ran out of image. Restart.'
            continue

        prev_patch = patch
        patch = bilinearInterpolation(img, grid_vectors_abs)

        sample_index += 1
        if debug:
            yield prev_patch, patch, grid_vectors_abs, img
        else:
            yield prev_patch, patch

        if sample_index == samples_per_file:
            sampleImg = True


def sequenceFromMovieData( mov, p, borderwidth=0, orientation='F', timelag=1 ):
    """
    Samples pxp patches from the 3D array mov with given timelag

    :param mov: 3D array with dimensions h x w x t

    """
    h, w, t = mov.shape

    max_h = h - p
    max_w = w - p
    max_t = t - timelag

    while True:
        xi = randint(low=0, high=max_w)
        yi = randint(low=0, high=max_h)
        ti = randint(low=0, high=max_t)
        X = mov[yi:yi+p,xi:xi+p,ti].flatten(orientation)
        Y = mov[yi:yi+p,xi:xi+p,ti+timelag].flatten(orientation)
        yield X, Y


def randShiftRotationScalingSequenceWithBorderIterator(dir, p, samples_per_file, loadfunc, borderwidth=0, orientation='F', shiftDistribution=None, rotationDistribution=None, scalingDistribution=None, upper_limit=2.0, lower_limit=0.5):
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
    elif shiftDistribution.param['n'] > 2:
        stdout.write('WARNING: shiftDistribution has more dimensions than needed. Using only first 2.\n')
    if rotationDistribution == None:
        raise ValueError, 'rotationDistribution cannot be None. Use e.g. Uniform(n=1, low=0.0, high=2*numpy.pi)'
    elif rotationDistribution.param['n'] > 1:
        stdout.write('WARNING: rotationDistribution has more dimensions than needed. Using only first.\n')
    if scalingDistribution == None:
        raise ValueError, 'scalingDistribution cannot be None. Use e.g. Uniform(n=1, low=0.0, high=360.0)'
    elif scalingDistribution.param['n'] > 2:
        stdout.write('WARNING: scalingDistribution has more dimensions than needed. Using only first 2.\n')

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
            width_ext = ceil(p * (sqrt(2)-1))
            w = p+2*width_ext
            width_limit = nx - abs(ceil(shift[-1])) - w - upper_limit*p
            height_limit = ny - abs(ceil(shift[0])) - w - upper_limit*p
            width_min = ceil(upper_limit/2.0*p)
            height_min = ceil(upper_limit/2.0*p)
            sampleImg = False
            stdout.write('\tSampling %i X %i patches from %s with shift ('%(p,p,filename) \
                         + shift.size*'%.3f, '%tuple(shift) + '\b\b)\n')
            stdout.flush()

        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0):
            xi = randint(low=width_min+offset[-1], high=width_limit)
            yi = randint(low=height_min+offset[0], high=height_limit)
            phi = rotationDistribution.sample(1).X[0,0]
            randSample = scalingDistribution.sample(1).X
            if randSample.size == 1:
                scale = array((randSample[0,0], randSample[0,0]))
            else:
                scale = array((randSample[0,0], randSample[1,0]))

            if scale[0] > upper_limit:
                scale[0] = upper_limit
            elif scale[0] < lower_limit:
                scale[0] = lower_limit
            if scale[1] > upper_limit:
                scale[1] = upper_limit
            elif scale[1] < lower_limit:
                scale[1] = lower_limit

            new_width = round(p*scale[0])
            new_height = round(p*scale[1])

            yinew = int((yi + p/2) - new_height/2 - width_ext)
            xinew = int((xi + p/2) - new_width/2 - width_ext)
            X = img[ yi:yi+p, xi:xi+p].flatten(orientation)

            if phi == 0.0:
                Ysource = img2[ yinew:yinew+new_height+2*width_ext, xinew:xinew+new_width+2*width_ext]
            else:
                Ysource = rotate( img2[ yinew:yinew+new_height+2*width_ext, xinew:xinew+new_width+2*width_ext], phi, reshape=False)
            #print Ysource.shape
            #print new_width, new_height, width_ext,
            if new_width == p and new_height == p:
                Y = Ysource[width_ext:new_height+width_ext,width_ext:new_width+width_ext].flatten(orientation)
            else:
                Y = zoom( Ysource[width_ext:new_height+width_ext,width_ext:new_width+width_ext], (p/new_height, p/new_width)).flatten(orientation)

            ptch = hstack((X.flatten(), Y.flatten()))

        sample_index += 1
        yield X, Y
        if sample_index == samples_per_file:
            sampleImg = True

    return

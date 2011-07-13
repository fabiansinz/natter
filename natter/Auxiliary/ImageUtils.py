from numpy import log,abs,real
from numpy import array as nArray
import array
import numpy as np
from numpy.random import randn, permutation
from numpy.fft import fft2,ifft2

def loadHaterenImage(filename):
    fin = open( filename, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('H', s)
    arr.byteswap()
    return log(nArray(arr, dtype='uint16').reshape(1024,1536))

def loadPhaseScrambledHaterenImage(filename):
    I = loadHaterenImage(filename)
    I2 = randn(I.shape[0],I.shape[1])
    fI2 = fft2(I2)
    fI = fft2(I)

    return real(ifft2(fI2/abs(fI2) * abs(fI)))

def loadAsciiImage(filename):
    f = open(filename,'r')
    X = []
    for l in f:
        X.append([float(elem) for elem in l.rstrip().lstrip().split()])
    f.close()
    return nArray(X)

def loadPixelScambledAsciiImage(filename):
    I = loadAsciiImage(filename)
    return I[permutation(I.shape[0]),permutation(I.shape[1])]

def loadReshadHaterenImage(filename):
    fin = open( filename, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('d', s)
    arr.byteswap()
    return nArray(arr, dtype='double').reshape(1024,1531)

def loadReshad2HaterenImage(filename):
    fin = open( filename, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('d', s)
    arr.byteswap()
    return nArray(arr, dtype='double').reshape(1021,1526)

def shiftImage( img, shift ):
    """shiftImage( img, shift )
    Calls shift1DImage( img, shift ) or shift2DImage( img, shift )
    depending on the dimension of the input image.
    """
    if img.ndim == 1:
        return shift1DImage(img, shift)
    elif img.ndim == 2:
        return shift2DImage(img, shift)
    else:
        raise ValueError('Neither 1D nor 2D image.')
    
def shift1DImage( img, shift ):
    """shift1DImage( img, shift )
    shifts 1D image by #shift pixels to the right by using Fourier shift theorem
    """
    if np.array(shift).size == 1:
        dx = -shift
        #if np.abs(dx) < 1:
        #    dx = dx * img.size
    else:
        raise ValueError('Shift is not a scalar.')

    IMG = np.fft.fft(img)
    n = IMG.size

    result = np.fft.ifft(IMG*np.exp(2J*np.pi*(np.fft.ifftshift(np.arange(0,n)-np.floor(n/2)))/n*dx))
    return result.real

def shift2DImage( img, shift ):
    """shift2DImage( img, shift )
    shifts 2D image by #shift[0] pixels vertically down and
    #shift[1] pixels horizontally to the right
    by using Fourier shift theorem
    """
    if np.array(shift).size == 1:
        dy = shift
        dx = shift
    elif np.array(shift).size == 2:
        dy, dx = shift           
    else:
        raise ValueError('Shift is not a scalar nor a 2x1 vector.')

    result = np.empty_like( img )
    [h, w] = img.shape
    for ii in xrange(h):
        result[ii,:] = shift1DImage(img[ii,:], dx)
    for ii in xrange(w):
        result[:,ii] = shift1DImage(result[:,ii], dy)
    
    return result

def bilinearInterpolation( img, vec, patch=None ):
    """bilinearInterpolation( img, vec, patch=None )
    computes the pixel values at all positions in vec (2xn matrix) by bilinear interpolation
    on img. If patch is given, result is written into patch. If patch is None or shape does
    not match a new matrix is created. In all cases, the result is also returned.

    :param img: Source image (2D, grayscale)
    :type img: numpy.ndarray
    :param vec: Array of 2D vectors at which pixel values are computed
    :type vec: numpy.ndarray
    :param patch: 1D array with same length as vec, will be filled with pixel values
    :type patch: numpy.ndarray
    :returns: Array of pixel values
    :rtype: numpy.ndarray
    """
    if patch is None:
        res = np.empty((vec.shape[1]))
    else:
        res = patch

    root = np.floor(vec)
    pos = vec - root
    dx = np.vstack((pos[0,:],1-pos[0,:]))
    dy = np.vstack((pos[1,:],1-pos[1,:]))
    for ii in xrange(res.size):
        win = img[root[1,ii]:root[1,ii]+2,root[0,ii]:root[0,ii]+2]
        res[ii] = np.dot(dx[:,ii], np.dot(win, dy[:,ii]))
    return res

def nearestNeighbor( img, vec, patch=None ):
    """nearestNeighbor( img, vec, patch=None )
    computes the pixel values at all positions in vec (2xn matrix) by nearest neighbor interpolation
    on img. If patch is given, result is written into patch. If patch is None or shape does
    not match a new matrix is created. In all cases, the result is also returned.

    :param img: Source image (2D, grayscale)
    :type img: numpy.ndarray
    :param vec: Array of 2D vectors at which pixel values are computed
    :type vec: numpy.ndarray
    :param patch: 1D array with same length as vec, will be filled with pixel values
    :type patch: numpy.ndarray
    :returns: Array of pixel values
    :rtype: numpy.ndarray
    """
    if patch is None:
        res = np.empty((vec.shape[1]))
    else:
        res = patch

    pos = np.round(vec)
    for ii in xrange(res.size):
        res[ii] = img[pos[1,ii],pos[0,ii]]
    return res

def bicubicInterpolation( img, vec, patch=None ):
    """bicubicInterpolation( img, vec, patch=None )
    computes the pixel values at all positions in vec (2xn matrix) by bicubic interpolation
    on img. If patch is given, result is written into patch. If patch is None or shape does
    not match a new matrix is created. In all cases, the result is also returned.

    :param img: Source image (2D, grayscale)
    :type img: numpy.ndarray
    :param vec: Array of 2D vectors at which pixel values are computed
    :type vec: numpy.ndarray
    :param patch: 1D array with same length as vec, will be filled with pixel values
    :type patch: numpy.ndarray
    :returns: Array of pixel values
    :rtype: numpy.ndarray
    """
    if patch is None:
        res = np.empty((vec.shape[1]))
    else:
        res = patch

    root = np.floor(vec)
    pos = vec - root
    dx = np.vstack((pos[0,:],1-pos[0,:]))
    dy = np.vstack((pos[1,:],1-pos[1,:]))
    grad = np.zeros(12)
    Ainv = np.array(((1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                     (0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),\
                     (-3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0),\
                     ( 2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),\
                     ( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),\
                     ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),\
                     ( 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0),\
                     ( 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0),\
                     ( -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0),\
                     ( 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0),\
                     ( 9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1),\
                     ( -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1),\
                     ( 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),\
                     ( 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0),\
                     ( -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1),\
                     ( 4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1)))
    
    for ii in xrange(res.size):
        win = img[root[1,ii]:root[1,ii]+2,root[0,ii]:root[0,ii]+2]
        grad[0] = img[root[1,ii],root[0,ii]+1]-img[root[1,ii],root[0,ii]]
        grad[2] = img[root[1,ii],root[0,ii]+2]-img[root[1,ii],root[0,ii]+1]
        grad[1] = img[root[1,ii]+1,root[0,ii]+1]-img[root[1,ii]+1,root[0,ii]]
        grad[3] = img[root[1,ii]+1,root[0,ii]+2]-img[root[1,ii]+1,root[0,ii]+1]
        grad[4] = img[root[1,ii]+1,root[0,ii]]-img[root[1,ii],root[0,ii]]
        grad[6] = img[root[1,ii]+2,root[0,ii]]-img[root[1,ii]+1,root[0,ii]]
        grad[5] = img[root[1,ii]+1,root[0,ii]+1]-img[root[1,ii],root[0,ii]+1]
        grad[7] = img[root[1,ii]+2,root[0,ii]+1]-img[root[1,ii]+1,root[0,ii]+1]
        grad[8] = grad[1]-grad[0]
        grad[9] = img[root[1,ii]+2,root[0,ii]+1]-img[root[1,ii]+2,root[0,ii]]-grad[1]
        grad[10] = grad[3]-grad[2]
        grad[11] = img[root[1,ii]+2,root[0,ii]+2]-img[root[1,ii]+2,root[0,ii]+1] - grad[3]
        x = np.hstack((win[0,:], win[1,:], grad))
        alpha = np.dot(Ainv, x)        
        res[ii] = 0.0
        for jj in xrange(3):
            for kk in xrange(3):
                res[ii] += alpha[jj*3+kk]*(dx[0,ii]**jj)*(dy[0,ii]**kk)
    return res

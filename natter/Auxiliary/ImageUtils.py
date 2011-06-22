import numpy
import array
import numpy as np

def loadHaterenImage(filename):
    fin = open( filename, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('H', s)
    arr.byteswap()
    return numpy.log(numpy.array(arr, dtype='uint16').reshape(1024,1536))

def loadReshadHaterenImage(filename):
    fin = open( filename, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('d', s)
    arr.byteswap()
    return numpy.array(arr, dtype='double').reshape(1024,1531)

def loadReshad2HaterenImage(filename):
    fin = open( filename, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('d', s)
    arr.byteswap()
    return numpy.array(arr, dtype='double').reshape(1021,1526)

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

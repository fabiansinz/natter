from numpy import log,abs,real
from numpy import array as nArray
import array
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


import numpy
import array

def loadHaterenImage(filename):
    fin = open( filename, 'rb' )
    s = fin.read()
    fin.close()
    arr = array.array('H', s)
    arr.byteswap()
    return numpy.log(numpy.array(arr, dtype='uint16').reshape(1024,1536))


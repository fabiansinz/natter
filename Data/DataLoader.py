import Data
from scipy import io
from scipy.sparse import lil_matrix
import pickle
from Auxiliary import Errors
from numpy import any, size, max, zeros, concatenate, shape, ndarray, array

def load(path):
    """
    DAT = LOAD(PATH)

    tries to load data from a specified file. If it cannot determine the file type, it throws an error.
    
    """
    pathjunks = path.split('.')
    if pathjunks[-1] == 'mat':
        return matlab(path)
    elif pathjunks[-1] == 'pydat':
        return pydat(path)
    elif pathjunks[-1] == 'dat':
        return ascii(path)

def matlab(path, varname=None):
    dat = io.loadmat(path,struct_as_record=True)
    if varname:
        return Data.Data(dat[varname],'Matlab data from ' + path)
    else:
        thekey = None
        maxdat = 0
        for k in dat.keys():
            if type(dat[k]) == ndarray:
                sh = shape(dat[k])
                if sh[0]*sh[1] > maxdat:
                    maxdat = sh[0]*sh[1]
                    thekey = k
        return Data.Data(dat[thekey],'Matlab variable ' + thekey + ' from ' + path)

def nisdetDataObject(path,varname='dat'):
    dat = io.loadmat(path,struct_as_record=True)[varname][0][0][1]
    return Data.Data(dat,'Data from NISDET data object ' + varname + ' from ' + path)

def ascii(path):
    f = open(path,'r')
    X = []
    for l in f:
        X.append([float(elem) for elem in l.rstrip().lstrip().split()])
    f.close()
    return Data.Data(array(X),'Ascii file read from ' + path)
    
def pydat(path):
    f = open(path,'r')
    dat = pickle.load(f)
    f.close()
    return dat

def libsvm(path,n=1):
    f = open(path,'r')
    L = f.readlines()
    f.close()
    m = len(L)
    X = zeros((m,n))
    i = 0
    for l in L:
        l = [e.split(':') for e in l.rstrip().lstrip().split()[1:]]
        ind = [int(e[0])-1 for e in l]
        val = [float(e[1]) for e in l]

        if any(ind <0):
            raise Errors.DataLoadingError('Index negative!')
        if max(ind) + 1 > n:
            X = concatenate((X,zeros((m,max(ind) + 1 - size(X,1)))),1)
            n = max(ind) + 1 - size(X,1)
            print ind
        X[i,ind] = val
        i += 1
    dat = Data.Data(X.transpose(),'Data from ' + path )
    dat.history.append('loaded from ' + path)
    dat.history.append('converted from libsvm format')
    return dat

     
    
    

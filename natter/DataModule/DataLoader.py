from sys import stdin, stdout
import gzip, bz2, zipfile
from natter.DataModule import Data
from scipy import io
import cPickle as pickle #cPickle bug seems to be fixed
#import pickle
from natter.Auxiliary import Errors
from numpy import any, size, max, zeros, concatenate, shape, ndarray, array, atleast_2d
import numpy as np

def load(path, **kwargs):
    """

    tries to load data from a specified file by determining the file
    type from its extension.

    *mat* are interpreted as matlab files. In this case it tries to
    load the largest variable. If you know the variables name, use the
    loading method *matlab* for which you can specify the variable
    name. Uses scipy.io and with scipy <=0.10 only compatible with matlab
    file format < v7.3.

    *dat* are interpreted as ascii files.

    *pydat* are interpreted as natter.DataModule.Data objects stored
     with pickle (because cPickle (binary file, latest protocol) caused
     MemoryErrors for files > 1GB on loading)

     *npz* are interpreted as numpy zip file. If the variable name is not
     given using varname parameter the user is asked to select one variable.

     *gz* are read as gzip compressed ascii file.

    :param path: Path to the data file.
    :type path: string
    :returns: Data object with the data from the specified file.
    :rtype: natter.DataModule.Data

    """
    pathjunks = path.split('.')
    if pathjunks[-1] == 'mat':
        return matlab(path, **kwargs)
    elif pathjunks[-1] == 'pydat':
        return pydat(path)
    elif pathjunks[-1] == 'dat':
        return ascii(path)
    elif pathjunks[-1] == 'npz':
        return loadnpz(path, **kwargs)
    elif pathjunks[-1] == 'gz':
        return ascii(path, open_file=gzip.open)
    else:
        raise ValueError('Could not determine file type. Please use one of the '+\
                         'supported file formats. See documentation for details.')

def matlab(path, varname=None):
    """
    Loads a matlab file from the specified path. If no variable name
    is passed to the function it uses the largest variable in the
    matlab file.

    :param path: Path to the .mat file.
    :type path: string
    :param varname: Name of the variable to be loaded from the .mat file.
    :type varname: string
    :returns: Data object with the data from the specified file.
    :rtype: natter.DataModule.Data

    """
    dat = io.loadmat(path,struct_as_record=True)
    if varname:
        return Data(dat[varname],'Matlab data from ' + path)
    else:
        thekey = None
        maxdat = 0
        for k in dat.keys():
            if type(dat[k]) == ndarray:
                sh = shape(dat[k])
                if sh[0]*sh[1] > maxdat:
                    maxdat = sh[0]*sh[1]
                    thekey = k
        return Data(dat[thekey],'Matlab variable ' + thekey + ' from ' + path)

def ascii(path, open_file=open):
    """
    Loads data from an ascii file.

    :param path: Path to the ascii file.
    :type path: string
    :returns: Data object with the data from the specified file.
    :rtype: natter.DataModule.Data

    """
    f = open_file(path,'r')
    X = []
    for l in f:
        X.append([float(elem) for elem in l.rstrip().lstrip().split()])
    f.close()
    return Data(array(X),'Ascii file read from ' + path)

def pydat(path):
    """
    Loads a natter Data object which was stored with pickle.

    :param path: Path to the .pydat file.
    :type path: string
    :returns: Data object with the data from the specified file.
    :rtype: natter.DataModule.Data

    """
    f = open(path,'rb')
    dat = pickle.load(f)
    f.close()
    return dat

def libsvm(path,n=1):
    """
    Loads data from a file in libsvm sparse format. The dimensionality
    of the data must be specified in advance.

    :param path: Path to the libsvm file.
    :type path: string
    :param n: Dimensionality of the data.
    :type n: int
    :returns: Data object with the data from the specified file.
    :rtype: natter.DataModule.Data

    """
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
    dat = Data(X.transpose(),'Data from ' + path )
    dat.history.append('loaded from ' + path)
    dat.history.append('converted from libsvm format')
    return dat

def loadnpz(path, varname=None, transpose=None):
    """
    Loads a npz file from the specified path. If no variable name
    is passed to the function it prints all variables and asks for
    user input.

    :param path: Path to the .npz file.
    :type path: string
    :param varname: Name of the variable to be loaded from the .npz file.
    :type varname: string
    :param transpose: Transpose of variable shall be loaded or the orientation shall be guessed
    :type transpose: bool
    :returns: Data object with the data from the specified file.
    :rtype: natter.DataModule.Data

    """
    fin = np.load(path)
    if varname is not None:
        if fin.keys().count(varname) > 0:
            dat = atleast_2d(fin[varname])
        else:
            raise ValueError, 'Given variable name "%s" does not exist in file "%s".'%(varname,path)

    else:
        stdout.write('Variables in "%s":\n'%(path))
        for var in fin.keys():
            stdout.write(var+'\n')
        stdout.write('Which variable should be loaded: ')
        var = stdin.readline()[:-1]
        if fin.keys().count(var) > 0:
            dat = atleast_2d(fin[var])
        else:
            raise ValueError, 'Given variable name "%s" does not exist in file "%s".'%(var,path)

    if transpose == None:
        if dat.shape[0] > dat.shape[1]:
            transpose = True
        else:
            transpose = False

    if transpose:
        return Data(dat.T,'npz data from ' + path)
    else:
        return Data(dat,'npz data from ' + path)

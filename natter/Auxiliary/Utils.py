import cPickle as pickle
import os
import cProfile
# import lsprofcalltree
from copy import deepcopy
from natter.Auxiliary.Errors import SpecificationError
from numpy.linalg import cholesky
from numpy import array, eye, dot, tile
from numpy.random import randn
try:
    import h5py
except:
    h5py = None

def parseParameters(args,kwargs):
    """
    Parses parameters and writes them into an dictionary. This
    function is heavily used by the __init__ function of the
    Distribution objects.

    :param args: args to a Distribution object
    :param kwargs: kwargs to a Distribution object
    :returns: parameter dict of the Distribution object
    :rtype: dictionary
    """
    param = None
    if len(args) == 1:
        param = args[0]
    if len(args) > 1:
        raise SpecificationError('*args may at most have length one!')
    if kwargs.has_key('param'):
        if param == None:
            param = kwargs['param']
        else:
            for k,v in kwargs['param'].items():
                param[k] = v
    if len(kwargs)>0:
        if param == None:
            param = kwargs
        else:
            for k,v in kwargs.items():
                if k != 'param':
                    param[k] = v
    return param

def save(o,filename):
    """
    Saves object o to file filename via pickle. If filename does not
    have an extension, .pydat is added.
    Uses highest available pickle protocol version

    :param o: pickleable object to save
    :type o: pickleable object
    :param filename: name of the file
    :type filename: string

    """
    tmp = filename.split('.')
    if tmp[-1] == 'pydat':
        f = open(filename,'w')
    else:
        f = open(filename + '.pydat','w')

    pickle.dump(o,f,pickle.HIGHEST_PROTOCOL)
    f.close()

def savehdf5( dat, filename ):
    """savehdf5( dat, filename )
    Saves the Data object in hdf5 format. Requires h5py and hdf5>1.8.3 installed.
    Adds .hdf5 to the end of the filename if filename does not end on .hdf5

    :param dat: natter Data object
    :type dat: natter.DataModule.Data
    :param filename: name of the file
    :type filename: string
    """
    if h5py is None:
        raise NotImplementedError('The module h5py could not be loaded, savehdf is therefore not available.')

    if filename.split('.')[-1].lower() != 'hdf5':
        filename += '.hdf5'

    fout = h5py.File(filename, 'w')
    fout.create_dataset('X', data=dat.X)
    fout.create_dataset('name', data=dat.name)
    hist = fout.create_group('history')
    saveListToHDF5(hist, dat.history, 0)
    fout.close()

def saveListToHDF5(fout, lst, counter):
    """saveListToHDF5(fout, lst, couter)
    Helper function to recursively save the embedded lists of the Data history
    to the hdf5 file in a recoverable way.

    :param fout: h5py stream (writable)
    :type fout: h5py.File or hpy5.Group object
    :param lst: list object which shall be stored
    :type lst: list
    :param counter: Counter of the history object, to keep order
    :type counter: int
    :returns: New counter
    ;rtype: int
    """
    for item in lst:
        if type(item) == str:
            fout.create_dataset('%i'%(counter), data=item)
        elif type(item) == list:
            grp = fout.create_group('%i'%(counter))
            counter = saveListToHDF5(grp, item, counter+1)
        else:
            print 'Unknown data type "%s" found. Casting to string.'%(type(item))
            fout.create_dataset('%i'%(counter), data=str(item))
        counter += 1
    return counter

def prettyPrintDict(value):
    """
    Returns a nice representation of a dictionary.

    :param value: dictionary to transform into a string
    :type value: dictionary
    :returns: string representation of the dictionary 'key : value' separated by lines
    :rtype: string
    """
    s = "\n"
    s+= 40*"=" + "\n"
    for (k,v) in value.items():
        s += str(k).upper()  + ": \n"
        s += str(v) + '\n'
        s += 40*'-' + '\n'
    s+= 40*"=" + "\n"
    return s




HaveIpython=True
try:
    from IPython.Debugger import Tracer
except:
    HaveIpython=False
    pass


def debug():
    """
    Invokes a debugger at the point where it is called, but only if
    the Ipython-shell is available.
    """

    if HaveIpython:
        Tracer()
    else:
        return





def profileFunction(f):
    """
    profiles the execution of a function via lsprofcalltree for later
    inspection with kcachegrind. kcachegrind is directly called with
    the corresponding profile.
    NOTE: this only works for linux systems where kcachegrind is installed.

    :param f: function handle of the function to profile.
    :type f: python-function

    """
    filename = '/tmp/profile.prof'
    p = cProfile.Profile()
    p.runcall(f)
    k = lsprofcalltree.KCacheGrind(p)
    data = open(filename, 'w+')
    k.output(data)
    data.close()
    cmd = "kcachegrind %s" % filename
    os.system(cmd)


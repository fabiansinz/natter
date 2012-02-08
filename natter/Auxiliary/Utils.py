import pickle
import os
import cProfile
# import lsprofcalltree
from copy import deepcopy
from natter.Auxiliary.Errors import SpecificationError
from numpy.linalg import cholesky
from numpy import array, eye, dot, tile
from numpy.random import randn


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
        
    pickle.dump(o,f)
    f.close()

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


import pickle
import os
import cProfile
import lsprofcalltree
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
    Saves object o to file filename via pickle. If filename does not have an extension, .pydat is added.

    :param o: pickleable object
    :param filename: name of the file
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

    :param value: dictionary
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
    from IPython.Debugger import Tracer;  debug = Tracer()
except:
    HaveIpython=False
    def debug():
        pass
    pass




def mnorm(nsamples=None,mu=None,sig=None):
    """
    generate nsamples of a gaussian random variable with mean mu and covariance
    sig, if none is given 1 sample of a gaussian random variable with mean=0 and
    covariance=1 is generated. This should be faster than the previous one.
    """
    if mu==None:
        mean = array([0.0])
    else:
        mean = mu.flatten()
    if sig ==None:
        cov = eye(len(mean))
    else:
        cov = sig
    if cov.shape[0]!=cov.shape[1] or cov.shape[0]!=len(mean):
        raise IndexError,"Given Covariance has inappropiate dimensions!"
    if nsamples==None:
        N=1
    else:
        N=nsamples
    L = cholesky(cov)
    r= randn(len(mu.flatten()),N)
    n=dot(L,r) + tile(mu.reshape(len(mu),1),N)
    return n


def profileFunction(f):
    """
    profiles the execution of a function via lsprofcalltree for later inspection with kcachegrind.

    :arguments:
        f   : function handle of the function to profile.
        filename: string of the filename to write profie information to.
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


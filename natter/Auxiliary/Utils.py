import pickle
import os
import cProfile
import lsprofcalltree
from copy import deepcopy
from natter.Auxiliary.Errors import SpecificationError

def parseParameters(args,kwargs):
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
    tmp = filename.split('.')
    if tmp[-1] == 'pydat':
        f = open(filename,'w')
    else:
        f = open(filename + '.pydat','w')
        
    pickle.dump(o,f)
    f.close()

def prettyPrintDict(value):
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

def fillDict(default,dic):
    ret = deepcopy(default)
    if dic==None:
        return ret
    else:
        for key in default.keys():
            if key in dic.keys():
                ret[key]=dic[key]
    return ret

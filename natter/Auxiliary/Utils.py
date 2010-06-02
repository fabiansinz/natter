import pickle
import os
import cProfile
import lsprofcalltree
from copy import deepcopy

def save(o,filename):
    tmp = filename.split('.')
    if tmp[-1] == 'pydat':
        f = open(filename,'w')
    else:
        f = open(filename + '.pydat','w')
        
    pickle.dump(o,f)
    f.close()

def testProtocol(value):
    s = "\n"
    s+= "++++++++++++++++++++++++ Test Error Protocol ++++++++++++++++++++++++\n"
    for (k,v) in value.items():
        s += str(k).upper()  + ": \n"
        s += str(v) + '\n'
        s += 10*'- - ' + '\n'
    s += "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
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

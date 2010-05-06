from Errors import *
import Optimization
import Plotting
from LpNestedFunction import LpNestedFunction
import pickle
import Numerics
import Statistics
from Potential import Potential
import ImageUtils

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

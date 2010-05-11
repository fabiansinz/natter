import pickle
from  Auxiliary import Errors,save
import types

class Filter:
    '''
    FILTER representing abstract filters

    '''

    def __init__(self):
        raise Errors.AbstractError('Filter is only an abstract class!')

    def logDetJacobian(self):
        raise Errors.AbstractError('Abstract method logDetJacobian() not implemented in ' + self.name)
        

    def addToHistory(self,hi):
        self.history.append(hi)

    def __mul__(self,O):
        return self.apply(O)

    def apply(self):
        raise Errors.AbstractError('Abstract method apply() not implemented in ' + self.name)

    def __invert__(self):
        raise Errors.AbstractError('Abstract method __invert__ not implemented in ' + self.name)
            
    def save(self,filename):
        Auxiliary.save(self,filename)

def load(path):
    f = open(path,'r')
    ret = pickle.load(f)
    f.close()
    return ret

def displayHistoryRec(h,recDepth=0):
    s = ""
    for elem in h:
        if type(elem) == types.ListType:
            s += displayHistoryRec(elem,recDepth+1)
        else:
            s += recDepth*'  ' + '* ' + elem + '\n'
    return s

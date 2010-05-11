from Transforms import Transform, LinearTransform
import numpy as np
import string
from Auxiliary import Errors
import Data
import types
import copy

class NonlinearTransform(Transform):
    '''
    NONLINEARTRANSFORM class representing nonlinear transformations.

    Each object stores a function f representing the transformation.
    '''

    def __init__(self,f=None,name='Noname',history=None,logdetJ=None):
        if history == None:
            self.history = []
        else:
            self.history = history
        self.f = f
        self.name = name
        self.logdetJ = logdetJ


    def apply(self,O):
        if isinstance(O,Data.Data):
            # copy other history 
            tmp = list(O.history)
            tmp.append('Applied non-linear filter "' + self.name + '"')
            tmp.append(list(self.history))

            # compute results and add own history
            ret = self.f(O)
            ret.history.append(tmp)
            return ret
        elif isinstance(O,LinearTransform.LinearTransform):
            # copy other history and add own
            tmp = list(O.history)
            tmp.append('multiplied with Transform "' + self.name + '"')
            tmp.append(list(self.history))

            Ocpy = O.copy()
            Scpy = self.copy()
            g = lambda x: Scpy.f(Ocpy.apply(x))
            gdet = None
            if Scpy.logdetJ != None:
                gdet = lambda y: self.logdetJ(Ocpy.apply(y)) + Ocpy.logDetJacobian()
            return NonlinearTransform(g,O.name,tmp, logdetJ=gdet )
        elif isinstance(O,NonlinearTransform):
            # copy other history and add own
            tmp = list(O.history)
            tmp.append('composed with "' + self.name + '"')
            tmp.append(list(self.history))

            Scpy = self.copy()
            Ocpy = O.copy()
            g = lambda x: Scpy.f( Ocpy.f(x) )
            gdet = None
            if self.logdetJ != None and O.logdetJ != None:
                gdet = lambda y: Scpy.logdetJ(Ocpy.f(y)) + Ocpy.logdetJ(y)
            return NonlinearTransform(g,O.name,tmp, logdetJ=gdet )
        else:
            raise TypeError('Transform.NoninearTransform.__mult__(): Transforms can only be multiplied with Data.Data, Transform.LinearTransform or Transform.NonlinearTransform objects')
        return self


    def logDetJacobian(self,dat):
        if self.logdetJ == None:
            raise Errors.AbstractError('logdetJ has not been specified!')
        else:
            return self.logdetJ(dat)

    def __call__(self,O):
        return self.apply(O)

    def __str__(self):
                
        s = 30*'-'
        s += '\nNonlinear Transform: ' + self.name + '\n'
        if len(self.history) > 0:
            s += Transform.displayHistoryRec(self.history,1)
        s += 30*'-'
        
        return s

    def getHistory(self):
        return list(self.history)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return copy.deepcopy(self)

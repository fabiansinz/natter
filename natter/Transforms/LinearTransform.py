import Transform
import NonlinearTransform
import string
from numpy.linalg import inv, det
from natter.Auxiliary import Errors, Plotting
from natter.DataModule import Data
import types
from numpy import array, ceil, sqrt, size, shape, concatenate, dot, log, abs

class LinearTransform(Transform.Transform):
    '''
    LINEARTRANSFORM class representing linear filters.

    Each object stores a numpy array W represening a array.
    '''

    def __init__(self,W=None,name='Noname',history=None):
        if history == None:
            self.history = []
        else:
            self.history = history

        self.W = array(W.copy())

        self.name = name

    def plotBasis(self):
        nx = ceil(sqrt(size(self.W,1)))
        ptchSz = sqrt(size(self.W,0))
        Plotting.plotPatches(inv(self.W),nx,ptchSz)

    def plotFilters(self):
        nx = ceil(sqrt(size(self.W,1)))
        ptchSz = sqrt(size(self.W,0))
        Plotting.plotPatches(self.W.transpose(),nx,ptchSz)

    def __invert__(self):
        sh = shape(self.W)
        if sh[0] == sh[1]:
            tmp = list(self.history)
            tmp.append('inverted')
            return LinearTransform(inv(self.W),self.name,tmp)
        else:
            raise Errors.DimensionalityError('Transform.__invert__(): Transform must be square!')

    def stack(self,F):
        tmp = self.getHistory()
        tmp.append('stacked with ' + F.name)
        return LinearTransform(concatenate((self.W,F.W),0),self.name,tmp)

    def transpose(self):
        tmp = list(self.history)
        tmp.append('transposed')
        return LinearTransform(array(self.W).transpose(),self.name,tmp)

    def apply(self,O):
        if isinstance(O,Data):

            # copy other history and add own 
            tmp = list(O.history)
            tmp.append('multiplied with Transform "' + self.name + '"')
            tmp.append(list(self.history))
            # compute function
            return Data(array(dot(self.W,O.X)),O.name,tmp)

        elif isinstance(O,LinearTransform):
            # cooy other history and add own
            tmp = list(O.history)
            tmp.append('multiplied with Transform "' + self.name + '"')
            tmp.append(list(self.history))

            # multiply both filters
            return LinearTransform(dot(self.W,O.W),O.name,tmp)
        
        elif isinstance(O,NonlinearTransform.NonlinearTransform):
            # copy other history and add own
            tmp = list(O.history)
            tmp.append('multiplied with Transform "' + self.name + '"')
            tmp.append(list(self.history))

            
            Ocpy = O.copy()
            Scpy = self.copy()
            g = lambda x: Scpy.apply(Ocpy.apply(x))
            gdet = None
            if Ocpy.logdetJ != None:
                gdet = lambda y: Ocpy.logdetJ(y) + Scpy.logDetJacobian()
            return NonlinearTransform.NonlinearTransform(g,Ocpy.name,tmp, logdetJ=gdet )
            
        else:
            raise TypeError('Transform.__mult__(): Transforms can only be multiplied with Data or Transform.LinearTransform objects')
            
        return self


    def det(self):
        return det(self.W)


    def logDetJacobian(self,dat=None):
        sh = shape(self.W)
        if sh[0] == sh[1]:
            if dat==None:
                return log(abs(det(self.W)))
            else:
                return array(dat.size(1)*[log(abs(det(self.W)))])
        else:
            raise Errors.DimensionalityError('Can only compute log det of square filter matrix')

    def __getitem__(self,key):
        tmp = list(self.history)
        tmp.append('subsampled')
        tmp2 = array(self.W[key])
        return LinearTransform(tmp2,self.name,tmp)


    def __str__(self):
        sh = string.join([str(elem) for elem in list(shape(self.W))],' X ')
        
        s = 30*'-'
        s += '\nLinear Transform (' + sh + '): ' + self.name + '\n'
        if len(self.history) > 0:
            s += Transform.displayHistoryRec(self.history,1)
        s += 30*'-'
        
        return s


    def getHistory(self):
        return list(self.history)

    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        return LinearTransform(self.W.copy(),self.name,list(self.history))


from Filter import Filter
from NonlinearFilter import NonlinearFilter
import string
from numpy.linalg import inv, det
from Auxiliary import Errors, Plotting
import Data
import types
from numpy import array, ceil, sqrt, size, shape, concatenate, dot, log

class LinearFilter(Filter):
    '''
    LINEARFILTER class representing linear filters.

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
            return LinearFilter(inv(self.W),self.name,tmp)
        else:
            raise Errors.DimensionalityError('Filter.LinearFilter.__invert__(): Filter must be square!')

    def stack(self,F):
        tmp = self.getHistory()
        tmp.append('stacked with ' + F.name)
        return LinearFilter(concatenate((self.W,F.W),0),self.name,tmp)

    def transpose(self):
        tmp = list(self.history)
        tmp.append('transposed')
        return LinearFilter(array(self.W).transpose(),self.name,tmp)

    def apply(self,O):
        if isinstance(O,Data.Data):

            # copy other history and add own 
            tmp = list(O.history)
            tmp.append('multiplied with Filter "' + self.name + '"')
            tmp.append(list(self.history))
            # compute function
            return Data.Data(array(dot(self.W,O.X)),O.name,tmp)

        elif isinstance(O,LinearFilter):
            # cooy other history and add own
            tmp = list(O.history)
            tmp.append('multiplied with Filter "' + self.name + '"')
            tmp.append(list(self.history))

            # multiply both filters
            return LinearFilter(dot(self.W,O.W),O.name,tmp)
        
        elif isinstance(O,NonlinearFilter):
            # copy other history and add own
            tmp = list(O.history)
            tmp.append('multiplied with Filter "' + self.name + '"')
            tmp.append(list(self.history))

            
            Ocpy = O.copy()
            Scpy = self.copy()
            g = lambda x: Scpy.apply(Ocpy.apply(x))
            gdet = None
            if Ocpy.logdetJ != None:
                gdet = lambda y: Ocpy.logdetJ(y) + Scpy.logDetJacobian()
            return NonlinearFilter(g,Ocpy.name,tmp, logdetJ=gdet )
            
        else:
            raise TypeError('Filter.LinearFilter.__mult__(): Filters can only be multiplied with Data.Data or Filter.LinearFilter objects')
            
        return self


    def det(self):
        return det(self.W)

    def logDetJacobian(self,dat=None):
        sh = shape(self.W)
        if sh[0] == sh[1]:
            if dat==None:
                return log(det(self.W))
            else:
                return array(dat.size(1)*[log(det(self.W))])
        else:
            raise Errors.DimensionalityError('Can only compute log det of square filter matrix')

    def __getitem__(self,key):
        tmp = list(self.history)
        tmp.append('subsampled')
        tmp2 = array(self.W[key])
        return LinearFilter(tmp2,self.name,tmp)


    def __str__(self):
        sh = string.join([str(elem) for elem in list(shape(self.W))],' X ')
        
        s = 30*'-'
        s += '\nLinear Filter (' + sh + '): ' + self.name + '\n'
        if len(self.history) > 0:
            s += Filter.displayHistoryRec(self.history,1)
        s += 30*'-'
        
        return s


    def getHistory(self):
        return list(self.history)

    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        return LinearFilter(self.W.copy(),self.name,list(self.history))


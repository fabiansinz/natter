import Transform
import NonlinearTransform
import string
from numpy.linalg import inv, det
from natter.Auxiliary import Errors, Plotting
from natter.DataModule import Data
from numpy import array, ceil, sqrt, size, shape, concatenate, dot, log, abs, reshape
import types

class LinearTransform(Transform.Transform):
    """
    LinearTransform class

    Implements linear transforms of data.
    
    :param W: Matrix for initializing the linear transform.
    :type W: numpy.array 
    :param name: Name of the linear transform
    :type name: string
    :param history: List of previous operations on the linear transform.
    :type history: List of (lists of) strings.
    
        
    """


    def __init__(self,W=None,name='Noname',history=None):
        if history == None:
            self.history = []
        else:
            self.history = history

        self.W = array(W.copy())

        self.name = name

    def plotBasis(self):
        """

        Plots the columns of the inverse linear transform matrix
        W. Works only if the square-root of the number of columns of W
        is an integer.

        """
        nx = ceil(sqrt(size(self.W,1)))
        ptchSz = sqrt(size(self.W,0))
        Plotting.plotPatches(inv(self.W),nx,ptchSz,contrastenhancement=True)
    
    def plotFilters(self):
        """

        Plots the rows of the linear transform matrix W. Works only if
        the square-root of the number of rows of W is an integer.

        """
        nx = ceil(sqrt(size(self.W,1)))
        ptchSz = sqrt(size(self.W,0))
        Plotting.plotPatches(self.W.transpose(),nx,ptchSz,contrastenhancement=True)
        
    def __invert__(self):
        """
        Overloads the ~ operator. Returns a new LinearTransform object
        with the inverse of the linear transform matrix W.

        :returns: A new LinearTransform object representing the inverted matrix W.
        :rtype: natter.Transforms.LinearTransform
        
        """
        sh = shape(self.W)
        if sh[0] == sh[1]:
            tmp = list(self.history)
            tmp.append('inverted')
            return LinearTransform(inv(self.W),self.name,tmp)
        else:
            raise Errors.DimensionalityError('Transform.__invert__(): Transform must be square!')

    def stack(self,F):
        """
        Returns a new LinearTransform object that contains the
        vertically stacked matrices of this object and the specified
        LinearTransform F.

        :param F: LinearTransform object to be stacked below this object.
        :type F: natter.Transforms.LinearTransform
        :returns: A new LinearTransform object with the stacked matrices.
        :rtype: natter.Transforms.LinearTransform
        
        """
        tmp = self.getHistory()
        tmp.append('stacked with ' + F.name)
        return LinearTransform(concatenate((self.W,F.W),0),self.name,tmp)

    def transpose(self):
        """
        Returns a new LinearTransform object with the transposed matrix W.
        
        :returns: A new LinearTransform object representing the transposed matrix W.
        :rtype: natter.Transforms.LinearTransform
        
        """
        
        tmp = list(self.history)
        tmp.append('transposed')
        return LinearTransform(array(self.W).transpose(),self.name,tmp)

    def T(self):
        """
        See transpose().
        
        :returns: A new LinearTransform object representing the transposed matrix W.
        :rtype: natter.Transforms.LinearTransform
        
        """
        
        return self.transpose()

    def apply(self,O):
        """
        Applies the LinearTransform object to *O*. *O* can either be

        * a natter.Transforms.LinearTransform object
        * a natter.Transforms.NonlinearTransform object
        * a natter.DataModule.Data object

        It also updates the correct computation of the log-det-Jacobian.

        
        :param O: Object this LinearTransform is to be applied to.
        :type O: see above.
        :returns: A new Transform or Data object
        :rtype: Depends on the type of *O*
        
        """
        
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
        """
        Computes the determinant of the LinearTransform objects matrix W.

        :returns: det(W)
        :rtype: float
        """
        return det(self.W)


    def logDetJacobian(self,dat=None):
        """
        Computes the determinant of the logarithm of the Jacobians
        determinant for the linear transformation (which is in this
        case only the log-determinant of W). If *dat* is specified it
        returns as many copies of the log determinant as there are
        data points in *dat*.


        :param dat: Data for which the log-det-Jacobian is to be computed.
        :type dat: natter.DataModule.Data
        :returns: The log-det-Jacobian 
        :rtype: float (if dat=None) or numpy.array (if dat!=None)
        """
        
        sh = shape(self.W)
        if sh[0] == sh[1]:
            if dat==None:
                return log(abs(det(self.W)))
            else:
                return array(dat.size(1)*[log(abs(det(self.W)))])
        else:
            raise Errors.DimensionalityError('Can only compute log det of square filter matrix')

    def __getitem__(self,key):
        """
        Mimics the __getitem__ routine on W. The only exception is
        that calls F[1,:] still return 2D arrays.

        :returns: New LinearTransform object where the W is the results of the __getitem__ operation.
        :rtype: natter.Transform.LinearTransform
        """
        
        tmp = list(self.history)
        tmp.append('subsampled')
            
        tmp2 = array(self.W[key])
        if type(key[0]) == types.IntType:
            tmp2 = reshape(tmp2,(1,len(tmp2)))
        elif len(key) > 1 and type(key[1]) == types.IntType:
            tmp2 = reshape(tmp2,(len(tmp2),1))
            
        return LinearTransform(tmp2,self.name,tmp)


    def __str__(self):
        """
        Returns a string representation of the LinearTransform object.

        :returns: A string representation of the LinearTransform object.
        :rtype: string
        """
        sh = string.join([str(elem) for elem in list(shape(self.W))],' X ')
        
        s = 30*'-'
        s += '\nLinear Transform (' + sh + '): ' + self.name + '\n'
        if len(self.history) > 0:
            s += Transform.displayHistoryRec(self.history,1)
        s += 30*'-'
        
        return s


    def getHistory(self):
        """
        Returns the history of the object. The history is a list of
        (list of ...) strings that store the previous operations
        carried out on the object.

        :returns: The history.
        :rtype: list of (list of ...) strings
        """
        return list(self.history)

    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        """
        Makes a deep copy of the LinearTransform and returns it.

        :returns: A deep copy of the LinearTransform object.
        :rtype: natter.Transforms.LinearTransform
        """
        return LinearTransform(self.W.copy(),self.name,list(self.history))


import Transform
import NonlinearTransform
import string
from numpy.linalg import inv, det
from natter.Auxiliary import Errors, Plotting
from natter.DataModule import Data
from numpy import array, ceil, sqrt, size, shape, concatenate, dot, log, abs, reshape, arange, zeros, where, meshgrid, sum, exp, pi, real, prod, floor, zeros, sqrt
import types
from matplotlib.pyplot import text
from numpy.fft import fft2
from scipy.optimize import fmin_l_bfgs_b

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

    def plotBasis(self, plotNumbers=False):
        """

        Plots the columns of the inverse linear transform matrix
        W. Works only if the square-root of the number of columns of W
        is an integer.

        :param plotNumbers: Determines whether the index of the basis function should be plotted as well.
        :type plotNumbers: bool

        """
        nx = ceil(sqrt(size(self.W,1)))
        ptchSz = sqrt(size(self.W,0))
        Plotting.plotPatches(inv(self.W),nx,ptchSz,contrastenhancement=True)

        if plotNumbers:
            row = 0
            col = 0
            i = 0
            
            while i < self.W.shape[0]:
                text(col+ptchSz/4,row+ptchSz/4,str(i),color='r',fontweight='bold')
                if col > (nx-2)*ptchSz:
                    col = 0
                    row += ptchSz
                else:
                    col += ptchSz
                i += 1

    
    def plotFilters(self, plotNumbers=False):
        """

        Plots the rows of the linear transform matrix W. Works only if
        the square-root of the number of rows of W is an integer.

        :param plotNumbers: Determines whether the index of the basis function should be plotted as well.
        :type plotNumbers: bool

        """
        nx = ceil(sqrt(size(self.W,1)))
        ptchSz = sqrt(size(self.W,0))
        Plotting.plotPatches(self.W.transpose(),nx,ptchSz,contrastenhancement=True)

        if plotNumbers:
            row = 0
            col = 0
            i = 0
            
            while i < self.W.shape[0]:
                text(col+ptchSz/4,row+ptchSz/4,str(i),color='r',fontweight='bold')
                if col > (nx-2)*ptchSz:
                    col = 0
                    row += ptchSz
                else:
                    col += ptchSz
                i += 1

        
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

    def getFourierMaximum(self):
        """
        Returns the 2D Fourier frequency that has the stronges amplitude. 

        :returns:   Returns the 2D Fourier frequency that has the stronges amplitude. 
        :rtype: numpy.array
        """
        p = sqrt(self.W.shape[1])
        f = arange(0,ceil(p/2.0))
        w = zeros((2,self.W.shape[0]))
        nx,ny = meshgrid(arange(p),arange(p))
        for i in xrange(self.W.shape[0]):
            patch = reshape(self.W[i,:],(p ,p), order='F') # reshape into patch
            z = abs(fft2(patch))[:ceil(p/2.0),:ceil(p/2.0)] # get fourier amplitude spectrum
            a = max(z.flatten()) # get maximal amplitude
            a = where(z == a) # get index of maximal amplitude
            w[:,i] = array([f[a[1][0]],f[a[0][0]]])
            g = lambda x: gratingProjection(x,p,nx,ny,patch,False)
            gprime = lambda x: gratingProjection(x,p,nx,ny,patch,True)

            #=========== DEBUG GRADIENT CHECK====
            # h = 1e-8
            # wtmp = array(w[:,i])
            # wtmph = array(wtmp)
            # wtmph[0] += h
            # print (g(wtmph) - g(wtmp))/h
            # wtmp = array(w[:,i])
            # wtmph = array(wtmp)
            # wtmph[1] += h
            # print (g(wtmph) - g(wtmp))/h
            # print gprime(w[:,i])
            # raw_input()
            #======================================
            w[:,i] =  fmin_l_bfgs_b(g, array([f[a[1][0]],f[a[0][0]]]) , fprime=gprime, bounds=( 2*[(0,floor(p/2.0))]))[0]

        return w

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

def gratingProjection(omega,p,nx,ny,f, fprime):
    if not fprime:
        tmp = sum( ( exp(1j*2*pi/p * (omega[0]*nx + omega[1]*ny))*f ).flatten())
        return -real(tmp*tmp.conjugate()) / p**2
    else:
        # h = 1e-8
        # ret = zeros((2,))
        # for k in xrange(2):
        #     tmp = array(omega)
        #     tmp[k] += h
        #     ret[k] = (gratingProjection(tmp,p,nx,ny,f,False) - gratingProjection(omega,p,nx,ny,f,False))/h
        # return ret
        dnx = reshape(nx,(prod(nx.shape),1))-reshape(nx,(1,prod(nx.shape)))
        dny = reshape(ny,(prod(ny.shape),1))-reshape(ny,(1,prod(ny.shape)))
        pf = reshape(f,(prod(f.shape),1))*reshape(f,(1,prod(f.shape)))
        tmp = exp(1j*2*pi/p *( omega[0]*dnx + omega[1]*dny) ) * pf * 1j*2*pi/p
        return -real(array([sum( (tmp*dnx).flatten() ), sum( (tmp*dny).flatten() ) ]) / p**2.0)


import Transform
import NonlinearTransform
import string
import cPickle as pickle
import numpy
from numpy.linalg import inv, det
from natter.Auxiliary import Errors, Plotting, hdf5GroupToList
from natter.DataModule import Data
from numpy import array,  size, shape, concatenate, dot, log, abs, reshape, arange,  meshgrid, sum, exp, pi, real, prod, floor, zeros, vstack, argmax,  sqrt, ceil, ones, complex_, isscalar
import types
from matplotlib.pyplot import text
#from matplotlib import pyplot
from scipy.optimize import fmin_l_bfgs_b
# from scipy.signal import hanning
from sys import stderr
try:
    import h5py
except:
    h5py = None

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

    def plotBasis(self, plotNumbers=False, orientation='F', **kwargs):
        """
        Plots the columns of the inverse linear transform matrix
        W. Works only if the square-root of the number of columns of W
        is an integer.

        :param plotNumbers: Determines whether the index of the basis function should be plotted as well.
        :type plotNumbers: bool
        :param orientation: matlab style column major ('F', default) or C/Python style row major ('C') reshaping, or no reshaping/1D plotting ('1D')
        :type orientation: string
        :param kwargs: See natter.Auxiliary.Plotting.plotStripes

        """
        if orientation == '1D':
            Plotting.plotStripes(inv(self.W),plotNumbers=plotNumbers, **kwargs)
        else:
            nx = ceil(sqrt(size(self.W,1)))
            ptchSz = sqrt(size(self.W,0))
            Plotting.plotPatches(inv(self.W),nx,ptchSz,contrastenhancement=True, orientation=orientation)

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


    def plotFilters(self, plotNumbers=False, orientation='F', **kwargs):
        """
        Plots the rows of the linear transform matrix W. Works only if
        the square-root of the number of rows of W is an integer.

        :param plotNumbers: Determines whether the index of the basis function should be plotted as well.
        :type plotNumbers: bool
        :param orientation: matlab style column major ('F', default) or C/Python style row major ('C') reshaping, or no reshaping/1D plotting ('1D')
        :type orientation: string
        :param kwargs: See natter.Auxiliary.Plotting.plotStripes

        """
        if orientation == '1D':
            Plotting.plotStripes(self.W.transpose(), plotNumbers=plotNumbers, **kwargs)
        else:
            nx = ceil(sqrt(size(self.W,0)))
            ptchSz = sqrt(size(self.W,1))
            Plotting.plotPatches(self.W.transpose(),nx,ptchSz,contrastenhancement=True, orientation=orientation)

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

    @property
    def T(self):
        """
        See transpose().

        :returns: A new LinearTransform object representing the transposed matrix W.
        :rtype: natter.Transforms.LinearTransform

        """

        return self.transpose()

    def __call__(self):
        """
        Returns the transformation. Needed for compatibility with old T syntax.
        """
        return self

    @property
    def shape(self):
        """
        Returns the shape of the transformation.

        :returns: shape of transformation
        :rtype: tuple
        """
        return self.W.shape

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
            def g(x):
                old_hist = list(x.history)
                ret  = Scpy.apply(Ocpy.apply(x))
                ret.history = old_hist
                return ret
            gdet = None
            if Ocpy.logdetJ != None:
                gdet = lambda y: Ocpy.logdetJ(y) + Scpy.logDetJacobian()
            return NonlinearTransform.NonlinearTransform(g,Ocpy.name,tmp, logdetJ=gdet )
        elif isscalar(O):
            self.W *= O
            self.history.append('multiplied by %f'%(O))
        else:
            raise TypeError('Transform.__mult__(): Transforms can only be multiplied with scalars, Data or Transform objects')

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

    def html(self):
        """
        Returns an html representation of itself. This is required by
        LogToken which LinearTransform inherits from.

        :returns: html preprentation the LinearTransform object
        :rtype: string

        """
        s = "<table border=\"0\"rules=\"groups\" frame=\"box\">\n"
        s += "<thead><tr><td colspan=\"2\"><tt><b>Linear Transform (%s): %s</b></tt></td></tr></thead>\n" \
             % (string.join([str(elem) for elem in list(shape(self.W))],' X '),self.name,)
        s += "<tbody>"
        s += "<tr><td valign=\"top\"><tt>History: </tt></td><td><pre>"
        if len(self.history) > 0:
            s += Transform.displayHistoryRec(self.history,1)
        s += "</pre></td></tr></table>"
        return s

    def getOptimalOrientationAndFrequency(self,delta=.25,weight=False, weightings=None):
        """
        Computes the optimal orientation and spatial frequency for all
        filters (rows). This is done by oversampling the Fourier space
        and computing the maximal absolute response to

        :math:`\sum_{\boldsymbol n} \exp(2*\pi*i*\langle\boldsymbol \omega, \boldsymbol x_{\boldsymbol n} \rangle  ) f(\boldsymbol x_{\boldsymbol n})`

        :param delta: bin size for oversampling in the Fourier domain
        :type delta: float
        :param weight: Whether the filter is to be weighted with a Gaussian envelope function
        :type weight: bool
        :param weightings: Array that stores the envelope weighting functions if specified
        :type weighting: numpy.array

        :returns: The frequency vectors that gives the maximal responses.
        :rtype: numpy.array

        """
        stderr.write("\tComputing optimal frequency and orientation ")
        p = sqrt(self.W.shape[1])
        w = zeros((2,self.W.shape[0]))

        wx,wy = meshgrid(arange(-floor(p/2.0),floor(p/2.0)+delta/2.0,delta), arange(0,floor(p/2.0)+delta/2.0,delta) )
        W = vstack((wx.flatten('F'),wy.flatten('F')))

        nx,ny = meshgrid(arange(p),arange(p))
        F = zeros((W.shape[1],p**2),dtype=complex_)
        for i in xrange(W.shape[1]):
            tmp = exp(1j*2.0*pi/p * (W[0,i]*nx + W[1,i]*ny))
            F[i,:] = tmp.flatten('F')
        for i in xrange(self.W.shape[0]):
            stderr.write(".")
            # fit gaussian envelope to linear filter
            if weight:
                h = _fitGauss2Grating(reshape(array(self.W[i,:]),(p,p))).flatten('F')
            else:
                h = ones((p**2,))
            # store envelopefunction if necessary
            if weightings != None:
                weightings[i,:] = h
            tmp = abs(dot(F,self.W[i,:]*h)) # get the frequency responses


            maxResponse = argmax(tmp) # extract maximal response
            w[:,i] = W[:,maxResponse]


            patch = array(reshape(self.W[i,:],(p ,p), order='F')) # reshape into patch

            # refine estimate
            g = lambda x: _gratingProjection(x,p,nx,ny,patch,False)
            gprime = lambda x: _gratingProjection(x,p,nx,ny,patch,True)

            w[:,i] =  fmin_l_bfgs_b(g, array(w[:,i]) , fprime=gprime, bounds=( [(-floor(p/2.0),floor(p/2.0)),(0,floor(p/2.0))]))[0]

        stderr.write("\n")
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

    def min(self):
        """
        Returns the minimum value of the transformation matrix.

        :returns: minimum value
        :rtype: numpy.float
        """
        return self.W.min()

    def max(self):
        """
        Returns the maximum value of the transformation matrix.

        :returns: maximum value
        :rtype: numpy.float
        """
        return self.W.max()

    def __iadd__(self, value):
        """
        Adds given linear transformation or scalar to current transformation (``self += value``)

        :param value: linear transform with identical shape
        :type value: natter.Transforms.LinearTransform or scalar (int/float)
        :returns: linear transform on which function was called
        :rtype: natter.Transforms.LinearTransform

        """
        if isinstance(value, LinearTransform):
            self.W += value.W
            self.history.append('added transform')
            self.history.append(value.getHistory())
        elif isscalar(value):
            self.W += value
            self.history.append('added %f'%(value))
        else:
            raise NotImplementedError('Only adding scalars or LinearTransforms implemented yet.')

        return self

    def __add__(self, value):
        """
        Adds given linear transformation or scalar to copy of current transformation (``return = self + value``)

        :param value: linear transform with identical shape
        :type value: natter.Transforms.LinearTransform or scalar (int/float)
        :returns: sum of linear transforms
        :rtype: natter.Transforms.LinearTransform

        """
        result = self.copy()
        result += value
        return result

    def __isub__(self, value):
        """
        Subtracts given linear transformation or scalar from current transformation (``self -= value``)

        :param value: linear transform with identical shape
        :type value: natter.Transforms.LinearTransform or scalar (int/float)
        :returns: linear transform on which function was called
        :rtype: natter.Transforms.LinearTransform

        """
        if isinstance(value, LinearTransform):
            self.W -= value.W
            self.history.append('subtracted transform')
            self.history.append(value.getHistory())
        elif isscalar(value):
            self.W -= value
            self.history.append('subtracted %f'%(value))
        else:
            raise NotImplementedError('Only subtracting scalars or LinearTransforms implemented yet.')

        return self

    def __sub__(self, value):
        """
        Subtracts given linear transformation or scalar from copy of current transformation (``return = self - value``)

        :param value: linear transform with identical shape
        :type value: natter.Transforms.LinearTransform or scalar (int/float)
        :returns: sum of linear transforms
        :rtype: natter.Transforms.LinearTransform

        """
        result = self.copy()
        result -= value
        return result

    def __idiv__(self, value):
        """
        Divides current transformation by scalar (``self /= value``)

        :param value: Scalar number
        :type value: scalar (int/float)
        :returns: linear transform on which function was called
        :rtype: natter.Transforms.LinearTransform

        """
        if isscalar(value):
            self.W /= value
            self.history.append('divided by %f'%(value))
        else:
            raise NotImplementedError('Only dividing scalars implemented yet.')

        return self

    def __div__(self, value):
        """
        Divides current transformation by scalar (``self /= value``)

        :param value: Scalar number
        :type value: scalar (int/float)
        :returns: linear transform on which function was called
        :rtype: natter.Transforms.LinearTransform

        """
        result = self.copy()
        result /= value
        return result

    def __imult__(self, value):
        """
        Multiplies given linear transformation or scalar to current transformation (``self *= value``)
        Uses correct matrix multiplication, not point-wise multiplication.

        :param value: linear transform with identical shape
        :type value: natter.Transforms.LinearTransform or scalar (int/float)
        :returns: linear transform on which function was called
        :rtype: natter.Transforms.LinearTransform

        """
        if isscalar(value):
            self.W *= value
            self.history.append('multiplied by %f'%(value))
            result = self
        else:
            result = self.apply(value)
        return result

    def __mult__(self, value):
        """
        Multiplies given linear transformation or scalar to copy of current transformation (``return = self * value``)
        Uses correct matrix multiplication, not point-wise multiplication.

        :param value: linear transform with identical shape
        :type value: natter.Transforms.LinearTransform or scalar (int/float)
        :returns: linear transform on which function was called
        :rtype: natter.Transforms.LinearTransform

        """
        result = self.copy()
        result *= value
        return result

    def reshape(self, *args, **kwargs):
        """
        Reshapes the LinearTransform, takes the same parameter as numpy.reshape

        :returns: Reshaped copy of given LinearTransform
        :rtype: natter.Transforms.LinearTransform

        """
        if kwargs.has_key('order'):
            order = kwargs['order']
        else:
            order = 'C'
        result = self.copy()
        result.W.reshape(*args, order=order)
        result.history.append('reshaped to {0!s} order {1}'.format(tuple(args), order))
        return result

    def inv(self):
        """
        Returns the inverse of the transform if it is square. Throws error otherwise.

        :returns: Inverse of the transform
        :rtype: natter.Transforms.LinearTransform
        """
        if self.W.shape[0] != self.W.shape[1]:
            raise ValueError('Linear transform must be square')
        result = self.copy()
        result.W = inv(result.W)
        result.history.append('inverted')
        return result

    @staticmethod
    def load(path):
        """
        Loads a saved LinearTransform object from the specified path.

        :param path: Path to the saved Transform object.
        :type path: string
        :returns: The loaded object.
        :rtype: natter.Transforms.LinearTransform
        """
        tmp = path.split('.')
        if tmp[-1] == 'pydat':
            f = open(path,'rb')
            ret = pickle.load(f)
            f.close()
        elif tmp[-1] == 'hdf5' and not h5py is None:
            fin = h5py.File(path, 'r')
            history = hdf5GroupToList(fin['history'], 0)[0]
            ret = LinearTransform(W=fin['W'][...], name=str(fin['name'][...]), history=history)
        else:
            print "Unknown file type. Trying pydat format."
            try:
                f = open(path,'rb')
                ret = pickle.load(f)
                f.close()
            except Exception, e :
                print "Loading failed with exception ", e.message
                ret = None
        return ret

def _gratingProjection(omega,p,nx,ny,f, fprime):
    if not fprime:
        tmp = sum( ( exp(1j*2*pi/p * (omega[0]*nx + omega[1]*ny))*f ).flatten())
        return -real(tmp*tmp.conjugate()) / p**2
    else:
        # h = 1e-8
        # ret = zeros((2,))
        # for k in xrange(2):
        #     tmp = array(omega)
        #     tmp[k] += h
        #     ret[k] = (_gratingProjection(tmp,p,nx,ny,f,False) - _gratingProjection(omega,p,nx,ny,f,False))/h
        # return ret
        dnx = reshape(nx,(prod(nx.shape),1))-reshape(nx,(1,prod(nx.shape)))
        dny = reshape(ny,(prod(ny.shape),1))-reshape(ny,(1,prod(ny.shape)))
        pf = reshape(f,(prod(f.shape),1))*reshape(f,(1,prod(f.shape)))
        tmp = exp(1j*2*pi/p *( omega[0]*dnx + omega[1]*dny) ) * pf * 1j*2*pi/p
        return -real(array([sum( (tmp*dnx).flatten() ), sum( (tmp*dny).flatten() ) ]) / p**2.0)


def _fitGauss2Grating(w):
    nx,ny = meshgrid(arange(w.shape[0]), arange(w.shape[1]))
    w = abs(w).flatten('F')
    w = w/sum(w)
    # pyplot.imshow(reshape(array(w),(17,17),order='F'),interpolation='nearest',cmap=pyplot.cm.gray)
    # pyplot.show()
    # raw_input()
    N = vstack((nx.flatten('F'),ny.flatten('F')))
    mu = reshape(dot(N,w),(2,1),order='F')
    C = dot(N-mu,(N-mu).T*reshape(array(w),(w.shape[0],1),order='F'))
    N = N[[1,0],:] # account for image coordinates
    # tmp = reshape(((2*pi)*sqrt(det(C)))**-1*exp(-0.5*sum((dot(inv(C),N-mu)*(N-mu))**2,0) ),nx.shape,order='F')
    # pyplot.figure()
    # pyplot.imshow(tmp,interpolation='nearest',cmap=pyplot.cm.gray)
    # pyplot.show()
    # raw_input()

    return reshape(((2*pi)*sqrt(det(C)))**-1*exp(-0.5*sum((dot(inv(C),N-mu)*(N-mu))**2,0) ),nx.shape,order='F')


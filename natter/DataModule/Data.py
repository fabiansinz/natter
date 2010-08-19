from numpy import eye, array, shape, size, sum, abs, ndarray, mean, reshape, ceil, sqrt, var, cov, exp, log,sign, dot, hstack, savetxt
from  natter.Auxiliary import  Errors, Plotting, save
import matplotlib as mpl
import pylab as pl
from numpy.linalg import qr, svd
import types
import sys


class Data:
    """
    Data

    Class for storage of scalar and multi-variate Data. 
    
    :param X: Array that holds the Data points. Each column is a single Data sample. 
    :type X: numpy.array
    :param name: User specified name of the Data object. 
    :type name: string
    :param history: User specified history of X when creating a new Data object. *history* is a list of strings describing previous processing steps. Instead of string, the single list items can be list of strings as well. 
    :type history: list
        
    """
    def __init__(self,X=None,name='No Name',history = None):
        if X == None:
            X=array([])
        else:
            self.X = X
        self.name = name
        if history == None:
            self.history = []
        else:
            self.history = history

    def __str__(self):
        sh = shape(self.X)
        s = 30*'-'
        s += '\nData object: ' + self.name + '\n'

        if len(sh) > 1:
            s += '\t' + str(size(self.X,1)) + '  Examples\n'
            s += '\t' + str(size(self.X,0)) + '  Dimensions\n'
        else:
            s += '\t' + str(size(self.X,0)) + '  Examples\n'

        if len(self.history) > 0:
            s += '\nHistory:\n'
            s += displayHistoryRec(self.history,1)
            
        s += 30*'-' + '\n'
        return s

    def setHistory(self,h):
        """
        Sets a new history of the Data object.

        :param h: New history
        :type h: (recursive) list of strings
        """
        self.history = h

    def norm(self,p=2.0):
        """
        Computes the (Lp)-norm of the data samples.

        :param p: p for the Lp-norm. Default is p=2 (Euclidean norm)
        :type p: float
        :returns: A new Data object containing the norms.
        :rtype: natter.DataModule.Data
        """
        return Data(sum(abs(self.X)**p,0)**(1/p))

    def normalize(self,p=2.0):
        """
        Normalizes the data points.

        :param p: p of the Lp-norm that is used to normalize the data points.
        :type p: float
        """
        p = float(p)
        self.scale(1.0/self.norm(p).X)
        self.history[-1] = 'Normalized Data with p='+str(p)

    def __repr__(self):
        return self.__str__()

    def __pow__(self,a):
        h = list(self.history)
        h.append('Exponentiated with ' + str(a))
        return Data(self.X.copy()**float(a),self.name,h)

    def plot(self):
        """
        Plots a scatter plot of the data points. This method works only for two-dimensional data.

        :raises: natter.Auxiliary.Errors.DimensionalityError
        """
        if not len(self.X) == 2:
            raise Errors.DimensionalityError('Data object must have dimension 2 for plotting!')
        else:
            h = mpl.pyplot.scatter(self.X[0],self.X[1],color='black',s=.5)
            pl.show()
            return h

    def numex(self):
        """
        Returns the number of examples stored in the Data object.

        :returns: number of examples.
        :rtype: int
        """
        return size(self.X,1)

    def dim(self):
        """
        Returns the number of dimensions stored in the Data object.

        :returns: number of dimensions.
        :rtype: int
        """
        return size(self.X,0)
            
    def addToHistory(self,hi):
        """
        Adds a new item to the Data object's history.

        :param hi: New history item.
        :type hi: string or list of strings
        """
        self.history.append(hi)

    def scale(self,s):
        """
        Scales *X* with the array *s*. If *s* has the same dimensionality as the number of examples, each dimension gets scaled with *s*. If *s* has the same dimension as the number of dimensions, each example is scaled with *s*. If *s* has the same shape as *X*, *X* and *s* are simply multiplied.

        *s* can also be stored in a Data object.

        :param s: The scale factor
        :type s: numpy.array or natter.DataModule.Data
        """
        name = ''
        scaledwhat = ''

        if not (type(s) == ndarray): # then we assume that s is a data object
            name = s.name
            s = s.X
        else:
            name = 'array'
            
        sh = s.shape
        if len(sh) == 1 or sh[0] == 1 or sh[1] == 1:
            if sh[0] == self.X.shape[0]:
                s = reshape(s,(self.X.shape[0],1))
                scaledwhat = 'each dimension'
            elif sh[0] == self.X.shape[1]:
                s = reshape(s,(1,self.X.shape[1]))
                scaledwhat = 'each example'
            elif (sh[0] == 1 and sh[1] != self.X.shape[1]) or (sh[1] == 1 and sh[0] != self.X.shape[0]):
                raise Errors.DimensionalityError('Dimensionality of s must either be equal to the number of examples or the number of dimensions')
        elif sh[0] != self.X.shape[0] or sh[1] != self.X.shape[1]:
            raise Errors.DimensionalityError('Dimensions of s do not match!')
        else:
            scaledwhat = 'whole data'
        self.history.append('Scaled ' + scaledwhat +' with ' + name)

        self.X = self.X*s


    def scaleCopy(self,s,indices=None):
        """
        Scales single dimensions with *s*. *s* must either be a numpy.array or a Data object. The Data object is copied for scaling. The copy is returned.

        :param s: Scale factors
        :type s: numpy.array or natter.DataModule.Data
        :param indices: Indices into selected dimensions that are to be rescaled with *s*
        :type indices: list or tuple of int
        :returns: The copied and rescaled Data object.
        :rtype: natter.DataModule.Data
        """
        if indices == None:
            indices = range(self.size(0))
        ret = self.copy()
        if not (type(s) == ndarray): # then we assume that s is a data object
            ret.history.append('Scaled ' + str(len(indices)) + ' dimensions with ' + s.name)
            s = s.X
        else:
            ret.history.append('Scaled ' + str(len(indices)) + ' dimensions with array')
        for i in indices:
            ret.X[i] = ret.X[i]*s
        return ret

    def mean(self):
        """
        Computes the mean of the data points.

        :returns: mean
        :rtype: numpy.array
        """
        return mean(self.X,1)

    def __getitem__(self,key):
        tmp = list(self.history)
        tmp.append('subsampled')
        tmp2 = array(self.X[key])
        if type(key[0]) == types.IntType:
            tmp2 = reshape(tmp2,(1,len(tmp2)))
        elif type(key[1]) == types.IntType:
            tmp2 = reshape(tmp2,(len(tmp2),1))
            
        return Data(tmp2,self.name,tmp)
        
    def plotPatches(self,m=-1):
        """
        If the Data objects holds patches flattened into vectors (Fortran style), then plotPatches can plot those patches. If *m* is specified, only the first *m* patches are plotted.

        :param m: Number of patches to be plotted.
        :type m: int 
        """
        if m == -1:
            m = self.size(1)
        nx = ceil(sqrt(m))
        ptchSz = int(sqrt(size(self.X,0)))
        Plotting.plotPatches(self.X[:,:m],nx,ptchSz)

    def var(self):
        """
        Computes the marginal variance of the data points.

        :returns: The marginal variance
        :rtype: numpy.array
        """
        return var(self.X,1)

    def center(self,mu=None):
        """
        Centers the data points on the mean over samples and dimensions. The motivation for this is that patches of natural images are usually sampled randomly from images, which makes them have a stationary statistics. Therefore, the mean in each dimension should be the same and we get a better estimate if we sample over pixels and dimensions.

        :param mu: Mean over samples and dimensions.
        :type mu: float
        :returns: Mean over samples and dimensions.
        :rtype: float
        """
        if mu == None:
            mu = mean(mean(self.X,1))
        self.X -= mu
        self.history.append('Centered on mean ' + str(mu) + ' over samples and dimensions')
        return mu

    def makeWhiteningVolumeConserving(self,method='project',D=None):
        """
        Rescales the data such that whitening becomes volume
        conserving, i.e. such that each whitening matrix has
        determinant one.

        There are two methods how to do that. By default the data is
        rescaled such that whitning is volume conserving *after
        projecting out the DC component*, i.e. the whitening matrix of
        *dat2* computed like

        >>> P = natter.Transforms.FilterFactory.DCnonDC(dat)
        >>> dat2 = P[1:,:]*dat

        has determinant one. This option is carried out by setting *method* to 'project'

        If method='raw', whitening becomes volume conerving without
        projecting out the DC component.

        For more information about the rescaling see [Eichhorn2009]_.

        :param method: Rescaling method as described above.
        :type method: string
        :param D: Singular values of the covariance matrix (with or without projecting out the DC, depending on *method*). If *D* is specified the no covariance matrix is computed and D is used to recompute the rescaling coefficient. 
        :type D: numpy.array
        :returns: The singular values of the respective covariance matrix.
        :rtype: numpy.array
        
        """
        if not type(method) == types.StringType:
            if D == None:
                sys.stderr.write("Warning: method should be a string! Assume that you passed me D and used method='project'")
                D = method
            else:
                raise TypeError("method must be a string!")

        s = 1.0
        if method == 'project':
            if D == None:
                n = self.size(0)
                P = eye(n)
                P[:,0] = 1
                (Q,R) = qr(P)
                F = Q.T
                F = F[1:,:]
                C = cov(dot(F,self.X))
                (U,D,V) = svd(C)
                D = array([((D[i] > 1e-8) and D[i] or 0.0) for i in range(len(D))])

        if method == 'raw':
            if D == None:
                C = cov(self.X)
                (U,D,V) = svd(C)
                D = array([((D[i] > 1e-8) and D[i] or 0.0) for i in range(len(D))])

        s = exp(-.5/len(D)*sum(log(D)))

            
        self.X *= s
        self.history.append('made whitening volume conserving with method "' + method + '"')
        return D


    def cov(self):
        """
        Computes the covariance matrix of the data.

        :returns: The covariance matrix.
        :rtype: numpy.array
        """
        return cov(self.X)

    def dnormdx(self,p=2.0):
        """
        Computes the derivative of the Lp-norm (specified by p) w.r.t to the data points.

        :param p: p for the Lp-norm (default p=2; Euclidean)
        :type p: float
        :returns: The derivatives
        :rtype: numpy.array
        """
        
        p =float(p)
        r = (self.norm(p).X)**(1-p)
        x = array(self.X)
        for k in range(len(x)):
            x[k] = r*sign(x[k])*abs(x[k])**(p-1.0) 
        return x
        
    def size(self,dim = (0,1)):
        """
        Returns the size of the data matrix *X*. Works just like numpy.size.

        :param dim: Dimension for which the size is to be computed (=0 --> number of examples; =1 --> number of dimensions)
        :type dim: int or tuple of int
        :returns: The requested dimensionality
        :rtype: int or tuple of int
        """
        if not (type(dim) == int) and len(dim) == 2:
            sh = shape(self.X)
            if len(sh)<2:
                return (1,sh[0])
            else:
                return sh
        elif type(dim) == int:
            sh = shape(self.X)
            if len(sh) < 2:
                if dim==0:
                    return 1.0
                else:
                    return sh[0]
            else:
                return sh[dim]
        else:
            raise Errors.DimensionalityError('Data matrices cannot have more than two dimensions!')
        
    def copy(self):
        """
        Makes a deep copy of the Data object and returns it.

        :returns: A deep copy.
        :rtype: natter.DataModule.Data
        """
        return Data(self.X.copy(),self.name,list(self.history))

    def save(self,filename,format='pickle'):
        """
        Save the Data object to a file.

        :param filename: Filename
        :type filename: string 
        :param format: format in which the data is to be saved (default is 'pickle'). Other choices are 'ascii'
        :type format: string
        
        """
        if format=='pickle':
            save(self,filename)
        elif format=='ascii':
            savetxt(filename,self.X,'%.16e')
            
            


    def append(self,O):
        """
        Concatenates the Data object with another Data object *O*.

        :param O: The Data object this Data object is concatenated with.
        :type O: natter.DataModule.Data

        """
        h = list(O.history)
        self.X = hstack((self.X,O.X))
        self.history.append('Concatenated with data from \"' + O.name + '\"')
        self.history.append(h)

def displayHistoryRec(h,recDepth=0):
    s = ""
    for elem in h:
        if type(elem) == types.ListType:
            s += displayHistoryRec(elem,recDepth+1)
        else:
            s += recDepth*'  ' + '* '+ elem + '\n'
    return s

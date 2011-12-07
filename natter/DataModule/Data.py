from __future__ import division
from numpy import eye, array, shape, size, sum, abs, ndarray, mean, reshape, ceil, sqrt, var, cov, exp, log,sign, dot, hstack, savetxt, vstack, where, int64, split
from  natter.Auxiliary import  Errors, Plotting, save
from matplotlib.pyplot import scatter,text, figure
import pylab as pl
from numpy.linalg import qr, svd
import types
import sys
from natter.Logging.LogTokens import LogToken
from scipy.stats import kurtosis
from numpy.random import randint

class Data(LogToken):
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
            X = array(X) # copy X
            if len(X.shape) == 1:
                X = reshape(X,(1,X.shape[0]))
            self.X = X
        self.name = name
        if history == None:
            self.history = []
        else:
            self.history = history

    def ascii(self):
        return self.__str__()

    def html(self):
        s = "<table border=\"0\"rules=\"groups\" frame=\"box\">\n"
        s += "<thead><tr><td colspan=\"2\"><tt><b>Data: %s</b></tt></td></tr></thead>\n" % (self.name,)
        s += "<tbody>"
        s += "<tr><td><tt>Examples: </tt></td><td><tt>%i</tt></td></tr>" % (self.X.shape[1],)
        s += "<tr><td><tt>Dimensions: </tt></td><td><tt>%i</tt></td></tr>" % (self.X.shape[0],)
        s += "<tr><td valign=\"top\"><tt>History: </tt></td><td><pre>"
        if len(self.history) > 0:
            s += displayHistoryRec(self.history,1)
        s += "</pre></td></tr></table>"
        return s

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


    def rectify(self):
        self.X[where(self.X <0)] = 0.0

    def abs(self):
        return Data(abs(self.X),self.name,list(self.history + ['Absolute value taken']))

    def fade(self,dat,h):
        """
        Fades this dataset into the dataset dat via the mask
        h. The mask is a vector of the same dimension as one data
        sample in self or dat. For each example xi in this dataset and
        the corresponding example yi in the dataset dat, the operation

        :math:`z_{ij} = h_j \cdot x_{ij} + (1-h_j)\cdot y_{ij}`

        is carried out. For that reason h must have entries between
        zero and one.

        :param dat: other dataset which is faded into this one. It must have the same dimension as this dataset.
        :type dat: natter.DataModule.Data
        :param h: mask of the same dimension as a single vector in dat. All entries must be between zero and one. 
        :type h: numpy.array
        """

        if dat.size() != self.size():
            raise Errors.DimensionalityError('Dimensionalities of two datasets do not match!')
        if len(h.shape) < 2:
            h = reshape(h,(self.X.shape[0],1))

        self.X = self.X*h + (1-h)*dat.X
        self.addToHistory(['Faded with dataset %s with history' % (dat.name), list(dat.history)])

    def stack(self,dat):
        """
        Stacks the current dataset with a copy of the dataset dat. Both must
        have the same number of examples.
        

        :param dat: Other data object with the same number of examples
        :type dat: natter.DataModule.Data
        """

        if dat.numex() != self.numex():
            raise Errors.DimensionalityError('Number of examples of two datasets do not match!')
        
        self.X = vstack((self.X,dat.copy().X))
        
        self.addToHistory(['Stacked with dataset %s with history' % (dat.name), list(dat.history)])

        return self
     
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
        history  = list(self.history)
        history.append('%.2f-norm taken' % (p,))
        return Data(sum(abs(self.X)**p,axis=0)**(1.0/p),name=str(self.name),history=history)

    def normalize(self,p=2.0):
        """
        Normalizes the data points.

        :param p: p of the Lp-norm that is used to normalize the data points.
        :type p: float
        """
        p = float(p)
        self.scale(1.0/self.norm(p).X)
        self.history[-1] = 'Normalized Data with p='+str(p)
        return self
        
    def __repr__(self):
        return self.__str__()

    def __pow__(self,a):
        h = list(self.history)
        h.append('Exponentiated with ' + str(a))
        return Data(self.X.copy()**float(a),self.name,h)

    def __add__(self,o):
        result = self.copy()
        result.append(o)
        return result

    def __iadd__(self,o):
        self.append(o)
        return self
    

    def plot(self):
        """
        Plots a scatter plot of the data points. This method works only for two-dimensional data.

        :raises: natter.Auxiliary.Errors.DimensionalityError
        """
        if not len(self.X) == 2:
            raise Errors.DimensionalityError('Data object must have dimension 2 for plotting!')
        else:
            h = scatter(self.X[0],self.X[1],color='black',s=.5)
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
        return self


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
        elif type(key[1]) == types.IntType or type(key[1]) == int64:
            tmp2 = reshape(tmp2,(tmp2.shape[0],1))
        return Data(tmp2,self.name,tmp)
        
    def plotPatches(self,m=-1, plotNumbers=False, orientation='F', **kwargs):
        """
        If the Data objects holds patches flattened into vectors (Fortran style), then plotPatches can plot those patches. If *m* is specified, only the first *m* patches are plotted.

        :param m: Number of patches to be plotted.
        :type m: int 
        """
        fig = figure()
        if m == -1:
            m = self.size(1)
        if kwargs.has_key('nx'):
            nx = kwargs.pop('nx')
        else:
            nx = int(ceil(sqrt(m)))
        
        ptchSz = int(sqrt(size(self.X,0)))
        Plotting.plotPatches(self.X[:,:m], (nx, ceil(m/nx)), ptchSz, orientation=orientation, **kwargs)

        if plotNumbers:
            row = 0
            col = 0
            i = 0
            while i < m:
                text(col+ptchSz/4,row+ptchSz/4,str(i),color='r',fontweight='bold')
                if col > (nx-2)*ptchSz:
                    col = 0
                    row += ptchSz
                else:
                    col += ptchSz
                i += 1
        return fig

    def var(self):
        """
        Computes the marginal variance of the data points.

        :returns: The marginal variance
        :rtype: numpy.array
        """
        return var(self.X,1)

    def kurtosis(self,bias=False):
        """
        Computes the kurtosis for each marginal using the kurtosis
        estimator of scipy.stats.

        :param bias: If False, then the calculations are corrected for statistical bias.
        :type bias: Bool
        :returns: Marginal kurtoses
        :rtype: numpy.array
        
        """
        return kurtosis(self.X,axis=1,bias=bias)

    

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

    @property
    def shape(self):
        """
        Returns the shape of the data set.

        :returns: shape of data set
        :rtype: tuple
        """
        return self.X.shape

    def split( self, pieces, axis=1 ):
        """
        Splits the data set into #pieces pieces along axis #axis.
        By default along axis 1 (i.e. along the examples).
        Returns tuple of data sets. See numpy.split for details.
        
        :param pieces: Number of pieces to split data into
        :type pieces: int
        :param axis: Axis along which to split, default 1
        :type axis: int
        :returns: Splited data sets
        :rtype: Tuple of natter.DataModule.Data
        """
        result = list()
        datasets = split(self.X, pieces, axis)
        for ii in xrange(len(datasets)):
            splitset = Data(X=datasets[ii], name=self.name, history=list(self.history))
            splitset.history.append('Splitted into %i pieces, this is piece %d'%(len(datasets), ii+1))
            result.append(splitset)
        return result


    ############ Iterators ########################################
    def bootstrap(self,n,m):
        """
        Iterator that returns n datasets with m examples, that have
        been randomly drawn from that data.

        :param n: number of datasets to be bootstrapped.
        :type n: int
        :param m: number of samples per dataset
        :type m: int
        
        """
        ne = self.X.shape[1]
        for k in xrange(n):
            ind= randint(ne,size=(m,))
            yield self[:,ind]
        return 

def displayHistoryRec(h,recDepth=0):
    s = ""
    for elem in h:
        if type(elem) == types.ListType:
            s += (recDepth-1)*'   ' +  displayHistoryRec(elem,recDepth+1)
        else:
            s += recDepth*'   ' + ' |-' + elem + '\n'
    return s

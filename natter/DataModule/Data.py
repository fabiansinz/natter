from numpy import eye, array, shape, size, sum, abs, ndarray, mean, array, reshape, ceil, sqrt, var, cov, exp, log,sign, dot
from  Auxiliary import  Errors, Plotting, save
import matplotlib as mpl
import pylab as pl
from numpy.linalg import qr, svd
import types
import pickle


class Data:
    '''
    DATA class for data storage

    Class variables are Data.name and Data.X. X is supposed to be an array which holds the data points in its columns.
    '''
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
        self.history = h

    def norm(self,p=2.0):
        return Data(sum(abs(self.X)**p,0)**(1/p))

    def normalize(self,p=2.0):
        p = float(p)
        self.scale(1.0/self.norm(p).X)
        self.history[-1] = 'Normalized data with p='+str(p)

    def __repr__(self):
        return self.__str__()

    def __pow__(self,a):
        h = list(self.history)
        h.append('Exponentiated with ' + str(a))
        return Data(self.X.copy()**float(a),self.name,h)

    def plot(self):
        if not len(self.X) == 2:
            raise Errors.DimensionalityError('Data object must have dimension 2 for plotting!')
        else:
            h = mpl.pyplot.scatter(self.X[0],self.X[1],color='black',s=.5)
            pl.show()
            return h
            
    def addToHistory(self,hi):
        self.history.append(hi)

    def scale(self,s):
        if not (type(s) == ndarray): # then we assume that s is a data object
            self.history.append('Scaled with ' + s.name)
            s = s.X
        else:
            self.history.append('Scaled with array')
        for i in range(len(self.X)):
            self.X[i] = self.X[i]*s


    def scaleCopy(self,s,indices=None):
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
        if m == -1:
            m = self.size(1)
        nx = ceil(sqrt(m))
        ptchSz = int(sqrt(size(self.X,0)))
        Plotting.plotPatches(self.X[:,:m],nx,ptchSz)

    def var(self):
        return var(self.X,1)

    def center(self,mu=None):
        if mu == None:
            mu = mean(mean(self.X,1))
        self.X -= mu
        self.history.append('Centered on mean ' + str(mu) + ' over samples and dimensions')
        return mu

    def makeWhiteningVolumeConserving(self,method='project',D=None):
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

            s = exp(-.5/len(D)*sum(log(D)))
            self.X *= s
            self.history.append('made whitening volume conserving with method "' + method + '"')
        return D


    def cov(self):
        return cov(self.X)

    def dnormdx(self,p=2.0):
        p =float(p)
        r = (self.norm(p).X)**(1-p)
        x = array(self.X)
        for k in range(len(x)):
            x[k] = r*sign(x[k])*abs(x[k])**(p-1.0) 
        return x
        
    def size(self,dim = (0,1)):
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
        return Data(self.X.copy(),self.name,list(self.history))

    def save(self,filename):
        save(self,filename)

def displayHistoryRec(h,recDepth=0):
    s = ""
    for elem in h:
        if type(elem) == types.ListType:
            s += displayHistoryRec(elem,recDepth+1)
        else:
            s += recDepth*'  ' + '* '+ elem + '\n'
    return s

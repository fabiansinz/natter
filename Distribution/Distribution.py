from Auxiliary import Errors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import Data
import copy
import numpy as np
import pickle
import Auxiliary
import types

#
# DISTRIBUTION
#
# is the mother class of all Distribution objects. It implements
# certain abstract functions that raise the appropriate errors if
# called but not implemented on child classes.
#
# Apart from that it implements routines for saving, loading and
# displaying Distribution objects.
#
class Distribution:

    param = {}
    name = 'Abstract Distribution'
    primary = [] # contains the names of the primary parameters

    def __init__(self,param):
        self.param = {}
        

    def setParam(self,key,value):
        self.param[key] = value
    
    def loglik(self,dat):
        raise Errors.AbstractError('Abstract method loglik not implemented in ' + self.name)

    def sample(self,m):
        raise Errors.AbstractError('Abstract method sample not implemented in ' + self.name)

    def pdf(self,dat):
        if hasattr(self,'loglik'):
            return np.exp(self.loglik(dat))
        raise Errors.AbstractError('Abstract method p not implemented in ' + self.name)

    def cdf(self,dat):
        raise Errors.AbstractError('Abstract method cdf not implemented in ' + self.name)

    def ppf(self,dat):
        raise Errors.AbstractError('Abstract method ppf not implemented in ' + self.name)


    def dldx(self,dat):
        raise Errors.AbstractError('Abstract method dldx not implemented in ' + self.name)

    def dldxdtheta(self,dat):
        raise Errors.AbstractError('Abstract method dldxdtheta not implemented in ' + self.name)

    def dldx2(self,dat):
        raise Errors.AbstractError('Abstract method dldx2 not implemented in ' + self.name)

    def dldx2dtheta(self,dat):
        raise Errors.AbstractError('Abstract method dldx2dtheta not implemented in ' + self.name)

    def dldtheta(self,dat,which=(None,)):
        raise Errors.AbstractError('Abstract method dldtheta not implemented in ' + self.name)

    def primary2array(self):
        raise Errors.AbstractError('Abstract method primary2array not implemented in ' + self.name)

    def array2primary(self):
        raise Errors.AbstractError('Abstract method array2primary not implemented in ' + self.name)
        

    def estimate(self,dat,which=None):
        raise Errors.AbstractError('Abstract method estimate not implemented in ' + self.name)

    def score(self, param, dat, compute_derivative=False):
        """
        score(param,dat, compute_derivative=False)

        must exhibit the following behaviour:

        1) if compute_derivative ==False, it returns the value of the score function at param and dat
        2) if compute_derivative ==False, it returns the derivative w.r.t the primary parameters of the score function at param and dat
        
        """
        raise Errors.AbstractError('Abstract method score not implemented in ' + self.name)

    def copy(self):
        return copy.deepcopy(self)

    def all(self,dat):
        return -np.mean(self.loglik(dat)) / dat.size(0) / np.log(2)
    
    def __str__(self):
        s = 30*'-'
        s += '\n' + self.name + '\n'
        later = []
        for k in self.param.keys():
            if not (type(self.param[k]) == types.FloatType)  \
                   and not (type(self.param[k]) == type('dummy')) \
                   and not (type(self.param[k]) == np.float64) \
                   and not (type(self.param[k]) == np.float32) \
                   and not (type(self.param[k]) == np.float) \
                   and not (type(self.param[k]) == type(1)):
                later.append(k)
            else:
                s += '\t' + k + ': '
                ss = str(self.param[k])
                s += ss + '\n'
            
        for k in later:
            s += '\t' + k + ': '
            if type(self.param[k]) == types.ListType:
                if isinstance(self.param[k][0],Distribution):
                    ss = "list of %d \"%s\" objects" % (len(self.param[k]),self.param[k][0].name)
                else:
                    ss = "list of %d \"%s\" objects" % (len(self.param[k]),str(self.param[k][0]))
            else:
                ss = '\n' + str(self.param[k])
                ss = ss.replace('\n','\n\t')
            s += ss + '\n'
            
        s += 30*'-' + '\n'
        return s

    def __repr__(self):
        return self.__str__()


    def histogram(self,dat,cdf = False):

        sh = np.shape(dat.X)
        if len(sh) > 1 and sh[0] > 1:
            raise Errors.DimensionalityError('Cannont plot data with more than one dimension!')
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x =np.squeeze(dat.X)
        
        n, bins, patches = ax.hist(x, np.max(sh)/400, normed=1, facecolor='blue', alpha=0.8)

        bincenters = 0.5*(bins[1:]+bins[:-1])

        y = self.pdf( Data.Data(bincenters))
        l = ax.plot(bincenters, y, 'k--', linewidth=2)

        if hasattr(self,'cdf') and cdf:
            z = self.cdf( Data.Data(bincenters))
            l = ax.plot(bincenters, z, 'k.-', linewidth=2)
            plt.legend( ('p.d.f.','c.d.f.','Histogram') )
        else:
            plt.legend( ('p.d.f.','Histogram') )
       
        ax.set_xlabel('x')
        ax.set_ylabel('Probability')
        ax.set_xlim(np.min(x), np.max(x))
        ax.grid(True)

        plt.show()

    def save(self,filename):
        Auxiliary.save(self,filename)

def load(path):
    f = open(path,'r')
    ret = pickle.load(f)
    f.close()
    return ret

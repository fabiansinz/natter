from natter.Auxiliary import Errors,save
from natter.DataModule import Data
import copy
from numpy import exp, mean, log, float32, float64, float, shape, squeeze, max, min
import pickle
import types
import pylab as plt
import string

class Distribution:

    def __init__(self,param=None):
        if param!=None:
            self.param = param
        else:
            self.param = {}
        self.name = 'Abstract Distribution'
        self.primary = [] # contains the names of the primary parameters, i.e. those that are going to be fitted        
        

    def setParam(self,key,value):
        self.param[key] = value
    
    def loglik(self,dat):
        raise Errors.AbstractError('Abstract method loglik not implemented in ' + self.name)

    def sample(self,m):
        """

        Samples m samples from the current distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """
        
        raise Errors.AbstractError('Abstract method sample not implemented in ' + self.name)

    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points
        in dat by calling the function loglik.

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :raises: natter.Auxiliary.Errors.AbstractError if loglik of that distribution is not implemented.
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        if hasattr(self,'loglik'):
            return exp(self.loglik(dat))
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

    def dldtheta(self,dat):
        raise Errors.AbstractError('Abstract method dldtheta not implemented in ' + self.name)

    def primary2array(self):
        raise Errors.AbstractError('Abstract method primary2array not implemented in ' + self.name)

    def array2primary(self):
        raise Errors.AbstractError('Abstract method array2primary not implemented in ' + self.name)
        

    def estimate(self,dat):
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
        """
        Creates a deep copy of the distribution object.

        :returns: Deep copy of the distribution object.
        """
        return copy.deepcopy(self)

    def all(self,dat):
        """

        Computes the average log-loss in bits per component of the current distribution on the data points in dat.
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  The average log-loss in bits per component.
        :rtype:    float
        
        """        
        
        return -mean(self.loglik(dat)) / dat.size(0) / log(2)
    
    def __str__(self):
        s = 30*'-'
        s += '\n' + self.name + '\n'
        later = []
        for k in self.param.keys():
            if not (type(self.param[k]) == types.FloatType)  \
                   and not (type(self.param[k]) == type('dummy')) \
                   and not (type(self.param[k]) == float64) \
                   and not (type(self.param[k]) == float32) \
                   and not (type(self.param[k]) == float) \
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
                elif type(self.param[k][0]) == type.ListType:
                    ss = "list of %d lists" % (len(self.param[k]),)
                elif type(self.param[k][0]) == type.TupleType:
                    ss = "list of %d tuples" % (len(self.param[k]),)
                else:
                    ss = "list of %d \"%s\" objects" % (len(self.param[k]),str(self.param[k][0]))
            else:
                ss = '\n' + str(self.param[k])
                ss = ss.replace('\n','\n\t')
            s += ss + '\n'
        s += '\n\tPrimary Parameters:'
        s += '[' + string.join(self.primary,', ') + ']\n'
        s += 30*'-' + '\n'
        return s

    def __repr__(self):
        return self.__str__()


    def histogram(self,dat,cdf = False):

        sh = shape(dat.X)
        if len(sh) > 1 and sh[0] > 1:
            raise Errors.DimensionalityError('Cannont plot data with more than one dimension!')
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x =squeeze(dat.X)
        
        n, bins, patches = ax.hist(x, max(sh)/400, normed=1, facecolor='blue', alpha=0.8)

        bincenters = 0.5*(bins[1:]+bins[:-1])

        y = self.pdf( Data(bincenters))
        l = ax.plot(bincenters, y, 'k--', linewidth=2)

        if hasattr(self,'cdf') and cdf:
            z = self.cdf( Data(bincenters))
            l = ax.plot(bincenters, z, 'k.-', linewidth=2)
            plt.legend( ('p.d.f.','c.d.f.','Histogram') )
        else:
            plt.legend( ('p.d.f.','Histogram') )
       
        ax.set_xlabel('x')
        ax.set_ylabel('Probability')
        ax.set_xlim(min(x),max(x))
        ax.grid(True)

        plt.show()

    def save(self,filename):
        save(self,filename)


    def __call__(self,dat,pa=None):
        if pa == None:
            return self.loglik(dat)
        else:
            pold = self.primary2array()
            self.array2primary(pa)
            ret = self.loglik(dat)
            self.array2primary(pold)
            return ret
            

def load(path):
    f = open(path,'r')
    ret = pickle.load(f)
    f.close()
    return ret

from natter.Auxiliary import Errors,save
from natter.DataModule import Data
import copy
from numpy import exp, mean, log, float32, float64, float, shape, squeeze, max, min, abs
import pickle
import types
import pylab as plt
import string
from scipy.optimize import newton
class Distribution:

    def __init__(self, *args,**kwargs):
                # parse parameters correctly
        param = None
        if len(args) > 0:
            param = args[0]
        if kwargs.has_key('param'):
            if param == None:
                param = kwargs['param']
            else:
                for k,v in kwargs['param'].items():
                    param[k] = v
        if len(kwargs)>0:
            if param == None:
                param = kwargs
            else:
                for k,v in kwargs.items():
                    if k != 'param':
                        param[k] = v
        
        # set default parameters

        if param!=None:
            self.param = param
        else:
            self.param = {}
        self.name = 'Abstract Distribution'
        self.primary = [] # contains the names of the primary parameters, i.e. those that are going to be fitted        
    
    def loglik(self,dat):
        raise Errors.AbstractError('Abstract method loglik not implemented in ' + self.name)

    def __getitem__(self,key):
        if key in self.parameters('keys'):
            return self.parameters()[key]
        else:
            raise KeyError("Parameter %s not defined for %s" % (key,self.name))

    def __setitem__(self,key,value):
        if key in self.parameters('keys'):
            self.param[key] = value
        else:
            raise KeyError("Parameter %s not defined for %s" % (key,self.name))

    def parameters(self,keyval=None):
        raise Errors.AbstractError('Abstract method parameters not implemented in ' + self.name)
        

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

    def ppf(self,u,maxiter=400, tol = 1e-6):
        '''

        Evaluates the percent point function (i.e. the inverse c.d.f.)
        of the current distribution.

        This is the generic implementation of the p.p.f which uses
        Newton-Raphson to invert the c.d.f.

        NOTE:

        1) Your current distribution must implement a c.d.f. Otherwise
        this will not work.

        2) This works only for univariate distributions. Strange
        things will happen for multivariate distributions.
        
        :param u:  Points at which the p.p.f. will be computed.
        :type dat: numpy.array
        :returns:  Data object with the resulting points in the domain of this distribution. 
        :rtype:    natter.DataModule.Data
           
        '''
        
        dat = Data(0*u,'Function values of the p.p.f of %s' % (self.name,))
        iteration = 0
        ridge = 1e-2
        while iteration < maxiter and max(abs(u-self.cdf(dat))) > tol:
            iteration += 1
            dat.X = dat.X - (self.cdf(dat)-u)/ 2 /(self.pdf(dat) + 1e-2)
            ridge *= 0.5

        if max(abs(u-self.cdf(dat))) > tol:
            print "\tWARNING! natter.Distributions.Distribution: generic ppf did not converge!"
            print max(abs(u-self.cdf(dat)))
        return dat
        
        
        
        


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
        for k in self.parameters('keys'):
            if not (type(self[k]) == types.FloatType)  \
                   and not (type(self[k]) == type('dummy')) \
                   and not (type(self[k]) == float64) \
                   and not (type(self[k]) == float32) \
                   and not (type(self[k]) == float) \
                   and not (type(self[k]) == type(1)):
                later.append(k)
            else:
                s += '\t' + k + ': '
                ss = str(self[k])
                s += ss + '\n'
            
        for k in later:
            s += '\t' + k + ': '
            if type(self[k]) == types.ListType:
                if isinstance(self[k][0],Distribution):
                    ss = "list of %d \"%s\" objects" % (len(self[k]),self[k][0].name)
                elif type(self[k][0]) == types.ListType:
                    ss = "list of %d lists" % (len(self[k]),)
                elif type(self[k][0]) == types.TupleType:
                    ss = "list of %d tuples" % (len(self[k]),)
                else:
                    ss = "list of %d \"%s\" objects" % (len(self[k]),str(self[k][0]))
            else:
                ss = '\n' + str(self[k])
                ss = ss.replace('\n','\n\t')
            s += ss + '\n'
        s += '\n\tPrimary Parameters:'
        s += '[' + string.join(self.primary,', ') + ']\n'
        s += 30*'-' + '\n'
        return s

    def __repr__(self):
        return self.__str__()


    def histogram(self,dat,cdf = False, ax=None,plotlegend=True):

        sh = shape(dat.X)
        if len(sh) > 1 and sh[0] > 1:
            raise Errors.DimensionalityError('Cannont plot data with more than one dimension!')
    
        if ax == None:
            fig = plt.figure()
            ax = fig.add_axes([.05,.05,.9,.9])

        x =squeeze(dat.X)
        
        n, bins, patches = ax.hist(x, max(sh)/400, normed=1, facecolor='blue', alpha=0.8)

        bincenters = 0.5*(bins[1:]+bins[:-1])

        y = self.pdf( Data(bincenters))
        ax.plot(bincenters, y, 'k--', linewidth=2)

        if hasattr(self,'cdf') and cdf:
            z = self.cdf( Data(bincenters))
            ax.plot(bincenters, z, 'k.-', linewidth=2)
            if plotlegend:
                plt.legend( ('p.d.f.','c.d.f.','Histogram') )
        elif plotlegend:
            plt.legend( ('p.d.f.','Histogram') )
       
        ax.set_xlabel('x')
        ax.set_ylabel('Probability')
        ax.set_xlim(min(x),max(x))
        ax.grid(True)

        #plt.show()

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

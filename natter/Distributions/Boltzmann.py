from numpy import zeros,ones,eye,sqrt,exp,log,dot,sum,array,where,tril,hstack,vstack,floor,ceil,diag
from natter.Auxiliary import fillDict
from numpy.random import randn,shuffle,rand
from natter.DataModule import Data
from natter.Auxiliary.Numerics import logsumexp
from natter.Auxiliary.Errors import NotImplementedError
from scipy import weave
from scipy.weave import converters
from scipy import optimize
from Distribution import Distribution
from copy import deepcopy

class Boltzmann(Distribution):
    """
    """

    def __init__(self, *args,**kwargs):
        """
        Constructor.

        The constructor is either called with a dictionary, holding
        the parameters (see below) or directly with the parameter
        assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
        also possible.

        :param param: Dictionary of parameter entries, possible keys are:
                      'L' : lower traingular part of connection matrix
                      'h' : bias term
                      'n' : number of nodes (dimensionality of b)
                      'logZ': log partition function
        """

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
        
        Distribution.__init__(self)
        if param==None:
            param = {'n':2}
        if 'n' in param.keys():
            self.param['n']=param['n']
        if 'L' in param.keys():
            self.param['L']= param['L']
            if self.param['L'].shape[0]!=self.param['n']:
                raise ValueError,"dimensionality of connection matrix does not match n!"
        else:
            self.param['L'] = zeros((self.param['n'],self.param['n']))
        if 'h' in param.keys():
            self.param['h']= param['h']
            if len(self.param['h'])!=self.param['n']:
                raise ValueError,"dimensionality of connection matrix does not match n!"
        else:
            self.param['h']=zeros(self.param['n'])
        if 'logZ' in param.keys():
            self.param['logZ'] = param['logZ']
        else:
            self.param['logZ'] = sum( log(exp(self.param['h']) + 1.0))
        self.name = "Boltzmann distribution with " + str(self.param['n']) + " binary nodes"
        self.primary = ['L','h']        # by default we only want to fit the
                                        # connection matrix and the bias term
                                        
        self.trilInd = where(tril(ones(self.param['L'].shape),-1)>0)

    def parameters(self,keyval=None):
        """

        Returns the parameters of the distribution as dictionary. This
        dictionary can be used to initialize a new distribution of the
        same type. If *keyval* is set, only the keys or the values of
        this dictionary can be returned (see below). The keys can be
        used to find out which parameters can be accessed via the
        __getitem__ and __setitem__ methods.

        :param keyval: Indicates whether only the keys or the values of the parameter dictionary shall be returned. If keyval=='keys', then only the keys are returned, if keyval=='values' only the values are returned.
        :type keyval: string
        :returns:  A dictionary containing the parameters of the distribution. If keyval is set, a list is returned. 
        :rtype: dict or list
           
        """
        if keyval == None:
            return deepcopy(self.param)
        elif keyval== 'keys':
            return self.param.keys()
        elif keyval == 'values':
            return self.param.value()        

    def loglik(self,data):
        """
        Calculates the log probability of the given dataset.
        """
        n,m = data.size()
        W= self.param['L'] + self.param['L'].T
        chunksize = min(int(floor(1000/n)),m) # to limit memory use
        L = zeros(m)
        start = 0
        end = chunksize
        while start<m:
            L[start:end]=diag(dot(dot(data.X[:,start:end].T,W),data.X[:,start:end]))
            end = min(end+chunksize,m)
            start = start+chunksize
        L = L + dot(self.param['h'],data.X) - self.param['logZ']
        #L = sum(dot(self.param['L'].T,data.X)**2,axis=0) -self.param['logZ']
        return L

    def pdf(self,dat):
        return exp(self.loglik(dat))

    def primary2array(self):
        arr = array([])
        if 'L' in self.primary:
            arr = self.param['L'][self.trilInd]
        if 'h' in self.primary:
            arr = vstack((arr,self.param['h']))
        if 'logZ' in self.primary:
            arr = vstack((arr,self.param['logZ']))
        return arr


    def array2primary(self,arr):
        if 'L' in self.primary:
            self.param['L'][self.trilInd]=arr[0:len(self.param['L'][self.trilInd].flatten())]
            arr = arr[len(self.param['L'][self.trilInd].flatten())::]
        if 'h' in self.primary:
            self.param['h']=arr[0:len(self.param['h'])]
            arr = arr[len(self.param['h'])::]
        if 'logZ' in self.primary:
            self.param['logZ']= arr[0]
    

    def sample(self,m,sampleOpts=None):
        """
        Sample from the Boltzmann distributions.

        Different sampling methods may be specified. Each sampling
        method can have its own additionaly set of parameters
        specified by sampleOpts.

        :param m:
           Number of samples to draw
           
        :param method:
           Which method to use (default: 'Gibbs')
           
        :param sampleOpts:
           optional parameter specifying additional parameters for the sampling method.

           
        
        """
        n = self.param['n']
        X = zeros((n,m))
        defaultParam = {'method' : 'Gibbs',
                        'burnIn': 100, # number of inter samples to wait in the beginning
                        'indepGap':5 } # number of inter samples to wait between acutal samples
        param = fillDict(defaultParam,sampleOpts)
        if param['method'] == 'Gibbs':
            state = (randn(n)>0)*1
            ind = range(n)
            J = self.param['L'] +self.param['L'].T # get connection matrix
            for k in xrange(param['burnIn']):
                shuffle(ind)            # shuffle in place for random conditional
                for i in ind:
                    h = 1.0/(1.0+exp(-2.*dot(J[i,:],state) + self.param['h'][i]));
                    state[i] = 1.0*(rand(1)<h)
            for k in xrange(m):
                for l in xrange(param['indepGap']):
                    shuffle(ind)            # shuffle in place for random conditional
                    for i in ind:
                        h = 1.0/(1.0+exp(-2.*dot(J[i,:],state)+ self.param['h'][i]));
                        state[i] = 1.0*(rand(1)<h)
                X[:,k]=state
            return Data(X)

        
    def estimatePartitionFunction(self,methodParam=None):
        """
        Estimates the partition function of the Boltzmann
        distribution. By default this is done via brute force
        enumeration. Alternativ methods are: noiseContrastive, importanceSampling.

        :param methodParam:
           Dictionary of additional parameters specifying the method to be used for estiamteing the partition function.


        :Usage:
        >>>n=2
        >>>p = Boltzmann({'n':n})
        >>>param = {'method':'importanceSampling','proposal': p}
        >>>model.estimatePartitionFunction(methodParam=param)


        
        """
        n = self.param['n']
        defaultParam = {'method':'brute'}
        param = fillDict(defaultParam,methodParam)
        if param['method']=='brute':
            m=2**n
            states = ones((n,m))
            fac=1
            for k in xrange(n):
                states[k,:]=array([[0]*fac,[1]*fac]*(2**(n-k-1))).flatten()
                fac = fac*2
            self.param['logZ']=0.0
            logP = self.loglik(Data(states))
            self.param['logZ'] = logsumexp(logP)
        elif param['method']=='importance':
            # do importance sampling

            if 'sampler' in param.keys():
                sampler = param['sampler']
            else:
                sampler = Boltzmann({'n':self.param['n']})
            if 'nsamples' in param.keys():
                nsamples = param['nsamples']
            else:
                nsamples = 1000*self.param['n']**2
            self.param['logZ']=0.0

            X = sampler.sample(nsamples)
            logweights = self.loglik(X) - sampler.loglik(X)
            self.param['logZ'] = logsumexp(logweights) -log(nsamples)
        else:
            raise NotImplementedError, ""


    def getLXX(self,dat):
        """
        returns 2*L.T (X X.T) [indices] that is the gradient of
        sum_x  x.T L L.T x wrt L where L is lower triangular.

        """
        n,m = dat.size
        WXXT    = zeros((n,n,m))
        code = """
        for (int i=0;i<n;i++){ for (int j=0;j<n;j++){ for (int l=0;l<m;l++){ for (int u=0;u<n;u++){
        WXXT(i,j,l) += L(i,u)*X(u,l)*X(j,l);
        }}}}
        """
        X = dat.X;
        weave.inline(code,
                     ['L', 'X', 'WXXT', 'n','m'],
                     type_converters=converters.blitz,
                     compiler = 'gcc')

        WXXT = 2.*WXXT[self.trilInd[0],self.trilInd[1],:]
        return WXXT
        
        
    def estimate(self,data,methodParam=None):
        """
        Estimate the primary parameters of the Boltzmann
        distribution. By default this is done via brute force
        enumeration of all possible states.

        """
        defaultParam = {'method':'brute'}

        param = fillDict(defaultParam,methodParam)
        
        if param['method']=='brute':
            XD = data.X
            n = XD.shape[0]
            m = 2**n
            XB = ones((n,m))
            fac=1
            for k in xrange(n):
                XB[k,:]=array([[0]*fac,[1]*fac]*(2**(n-k-1))).flatten()
                fac = fac*2

            def logp(arr):
                self.array2primary(arr)
                bD = dot(self.L.T,XD)
                bB = dot(self.L.T,XB)
                logP = sum(bD**2,axis=0)
                logZ = logsumexp(sum(bB**2,axis=0))
                self.param['logZ']=logZ
                return -sum(logP-logZ)
            def fprime(arr):
                self.array2primary(arr)
                if 'L' in self.primary:
                    bB = dot(self.L.T,XB)
                    mD = self.getLXX(XD)
                    mM = self.getLXX(XB)
                    logP = sum(bB**2,axis=0)
                    mM = mM*exp(logP -logsumexp(logP))
                    grad = -sum(mD -mM,axis=1)
                if 'logZ' in self.primary:
                    if 'L' in self.primary:
                        grad = vstack((grad,array([float(m)])))
                    else:
                        grad = array([float(m)])
                return grad
            def check(arr):
                err = optimize.check_grad(logp,fprime,arr)
                print "Error in gradient : ", err
                
            arr0 = self.primary2array()
            arropt = optimize.fmin_bfgs(logp,arr0,fprime,callback=check)
        else:
            raise NotImplementedError('Method not implemented (yet).')
        
     
                
            
                
        

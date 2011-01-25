from Distribution import Distribution
from natter.DataModule import Data
from numpy import cumsum, array, log, pi, zeros, squeeze, Inf, floor, mean, exp, sum, dot, sqrt, abs, max
from numpy.random import rand, randn 
from scipy import stats
import sys
from natter.Auxiliary.Numerics import logsumexp
from copy import deepcopy


class MixtureOfGaussians(Distribution):
    """
    Mixture of Gaussians

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    
    :param param:
        dictionary which might containt parameters for the Mixture of Gaussians
              'K'    :   number of mixtures (default=3)

              's'    :   standard deviations  
   
              'mu'   :   means

              'pi'  :   prior mixture probabilities (default normalized random array of lenth L)

    :type param: dict

    Primary parameters are ['pi','mu','s'].
        
    """

    
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
        self.name = 'Mixture of Gaussians'
        self.param = {'K':3,'s':5.0*rand(3),'mu':10.0*randn(3),'pi':rand(3) }
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
            self.param['K'] = int(self.param['K'])
            if not param.has_key('s'):
                self.param['s'] = 5.0*rand(self.param['K'])
            if not param.has_key('mu'):
                self.param['mu'] = 10.0*randn(self.param['K'])
            if not param.has_key('pi'):
                self.param['pi'] = rand(self.param['K'])
        self.param['pi'] /= sum(self.param['pi'])
        self.primary = ['pi','mu','s']
                
                                            
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
        
    def sample(self,m):
        """

        Samples m samples from the current mixture of Gaussians.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        u = rand(m)
        cum_pi = cumsum(self.param['pi'])
        k = array(m*[0])
        for i in range(m):
            for j in range(len(cum_pi)):
                if u[i] < cum_pi[j]:
                    k[i] = j
                    break
        k = tuple(k)
        return Data(randn(m)*self.param['s'].take(k) + self.param['mu'].take(k),str(m) + ' sample from ' + self.name)
                

    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points,

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.pdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret


    def __kloglik(self,dat,k):
        return -.5*log(pi*2.0) - log(self.param['s'][k]) - (dat.X-self.param['mu'][k])**2 / (2.0*self.param['s'][k]**2.0)
        


    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''
        ret = zeros((self.param['K'],dat.size(1)))
        for k in range(self.param['K']):
            ret[k,:]  = log(self.param['pi'][k]) + squeeze(self.__kloglik(dat,k))
        return logsumexp(ret,0)




    def dldx(self,dat):
        '''
        Computes the derivative of the log-likelihood w.r.t. the data points in dat.

        :param dat: Data for which the derivatives will be computed.
        :type dat:  natter.DataModule.Data
        :returns:   Array containing the derivatives.
        :rtype:     numpy.array
        '''
        ret = 0.0*dat.X
        tmp = 0.0*dat.X
        for k in range(self.param['K']):
            tmp += self.param['pi'][k] * stats.norm.pdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
            ret +=  self.param['pi'][k] * stats.norm.pdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])\
                   * -self.param['s'][k]**(-2.0)  * (dat.X-self.param['mu'][k])
        ret /= tmp
        return ret


    def estimate(self,dat, errTol=1e-4,maxiter=1000):
        '''

        Estimates the parameters from the data in dat. It is possible to only selectively fit parameters of the distribution by setting the primary array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).

        The estimation method uses EM to fit the mixture distribution.
        
        :param dat: Data points on which the Mixture of Dirichlet distributions will be estimated.
        :type dat: natter.DataModule.Data
        :param errTol: Stopping criterion for the iteration
        :type method: float
        :param maxiter: maximal number of EM iterations
        :param maxiter: int
        
        '''
        
        print "\tEstimating Mixture of Gaussians with EM ..."
        errTol=1e-5




        K=self.param['K']
        mu = self.param['mu'].copy()
        s = self.param['s'].copy()
        m = dat.size(1)
        p = self.param['pi'].copy()
        X = dat.X
        
        H = zeros((K,m))
        ALLold = ALL = Inf

        nr = floor(m/K)
        for k in range(K):
            mu[k] = mean(X[k*nr:(k+1)*nr+1])

        for i in range(maxiter):
            ALLold = ALL
            sumH = zeros((1,m))
            for j in range(K):
                if p[j] < 1e-3:
                    p[j] = 1e-3
                if s[j] < 1e-3:
                    s[j] = 1e-3
            # E-Step
            # the next few lines have been transferred to the log-domain for numerical stability
            for k in range(K):
                H[k,:] = log(p[k]) + squeeze(-.5*log(pi*2.0) - log(s[k]) - (dat.X-mu[k])**2 / (2.0*s[k]**2.0)) 

            sumH = logsumexp(H,0)
            for k in range(K):
                H[k,:] = H[k,:] - sumH

            H = exp(H) # leave log-domain here
            sumHk = sum(H,1)


            if 'mu' in self.primary:
                mu = dot(H,X.transpose())/sumHk
                self.param['mu'] = mu
            if 'pi' in self.primary:
                p = squeeze(mean(H,1))
                self.param['pi'] = p
            if 's' in self.primary:
                for k in range(K):
                    s[k] = sqrt(sum(H[k,:]*(X-mu[k])**2)/sumHk[k])
                self.param['s'] = s
            if i >= 2:
                ALL = self.all(dat)
                print "\r\t Mixture Of Gaussians ALL: %.8f [Bits]" % ALL,
                sys.stdout.flush()
                if abs(ALLold-ALL)<errTol:
                    break
        print "\t[EM finished]"

        
        
    def cdf(self,dat):
        '''

        Evaluates the cumulative distribution  function on the data points,

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the quantiles.
        :rtype:    numpy.array
           
        '''

        
        ret = 0.0*dat.X
        for k in range(self.param['K']):
            ret += self.param['pi'][k] * stats.norm.cdf(dat.X,loc=self.param['mu'][k],scale=self.param['s'][k])
        return ret
        

        
    def ppf(self,u,maxiter=500, tol = 1e-5):
        '''

        Evaluates the percent point function (i.e. the inverse c.d.f.)
        of the mixture of Gaussians distribution.

        It uses a Newton-Raphson method with preinitialization.
        
        :param u:  Points at which the p.p.f. will be computed.
        :type dat: numpy.array
        :returns:  Data object with the resulting points in the domain of this distribution. 
        :rtype:    natter.DataModule.Data
           
        '''


        # preinitialization: if there was just a single Gaussian
        # weighted by pi_k, the cdf would saturize to pi_k, the cdf of
        # this Gaussians mean would lie at pi_k/2. If the Gaussians
        # were we separated, the cdf ranges would approximately split
        # up [0,1] in [0,pi_1,pi-1+pi_2, ..., 1]. We initialize the x
        # for each u with the mean of the Gaussian that corresponds to
        # that interval.

        print "\tpreinitialize ..."
        U = cumsum(self.param['pi'])
        X = 0*u
        m = max(u.shape)
        for i in xrange(m):
            k = 0
            while u[i] > U[k]:
                k +=1
            X[i] = self.param['mu'][k]
        
        
        
        dat = Data(X,'Function values of the p.p.f of %s' % (self.name,))
        iteration = 0
        sys.stderr.write("\tNewton-Raphson ...")
        while iteration < maxiter and max(abs(u-self.cdf(dat))) > tol:
            sys.stderr.write('%03i\b\b\b' % (iteration,))
            iteration += 1
            dat.X = dat.X - (self.cdf(dat)-u)/ 2 /(self.pdf(dat) + 1e-2)
        print ""
        if max(abs(u-self.cdf(dat))) > tol:
            print "\tWARNING! natter.Distributions.MixtureOfGaussians: ppf did not converge!"
            print max(abs(u-self.cdf(dat)))
        
        return dat
        
        

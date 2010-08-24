from Distribution import Distribution
from natter.DataModule import Data
from numpy import dot, log, sum, mean, ones, shape, abs, exp, where, max, median
from numpy.random import rand
from numpy.random.mtrand import dirichlet
from natter.Auxiliary import Errors
from natter.Auxiliary.Numerics import inv_digamma, digamma, trigamma
from scipy.special import gammaln
from copy import deepcopy

class Dirichlet(Distribution):
    """
    Dirichlet Distribution

    on the n-dimensional probability simplex.

    :param param:
        dictionary which might containt parameters for the Dirichlet
              'alpha'    :    alpha parameters (default = rand(10)). alpha also determines the dimensionality.

    :type param: dict

    Primary parameters are ['alpha'].
        
    """

    maxiter = 1000
    innermaxiter = 100
    tol = 1e-6
    innertol = 1e-6
    
    def __init__(self,param=None):
        """
        """
        
        self.name = 'Dirichlet Distribution'
        self.param = {'alpha':rand(10)}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        self.primary = ['alpha']
        
    def sample(self,m):
        """

        Samples m samples from the current Dirichlet distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """

        return Data(dirichlet(tuple(self.param['alpha']),m).transpose(),str(m) + ' samples from ' + self.name)
        
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

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        return dot(self.param['alpha']-1.0,log(dat.X)) \
             + gammaln(sum(self.param['alpha'])) - sum(gammaln(self.param['alpha']))
    
    
    def estimate(self,dat):
        """

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on
        the Distributions module <tutorial_Distributions>`).

        Estimate uses the algorithm as described in [Minka2000]_.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data

        This code was initially written for Matlab by Tom Minka.
        """


        bar_p = mean(log(dat.X),1)

        a = dirichletMomentMatch(dat.X)

        s = sum(a)

        if s <= 0.0:
            if s == 0.0:
                a = ones(shape(a))/float(len(a))
            else:
                a = a/s
            s = 1.0



        for i in xrange(self.maxiter):
            old_s = s

            a = dirichlet_fit_s(dat, a, bar_p, maxiter=self.innermaxiter, tol=self.innertol)
           
            s = sum(a)
            a = dirichlet_fit_m(dat, a, bar_p, 1, tol=self.innertol) 
            m = a/s
            if abs(s - old_s) < self.tol:
                break

        self.param['alpha'] = a

def dirichlet_fit_s(dat,a,bar_p=None,maxiter=100,tol=1e-6):
        """

        DIRICHLET_FIT_S   Maximum-likelihood Dirichlet precision.

        DIRICHLET_FIT_S(data,a) returns the MLE (a) for the matrix DATA,
        subject to a constraint on A/sum(A).
        Each row of DATA is a probability vector.
        A is a row vector providing the initial guess for the parameters.

        A is decomposed into S*M, where M is a vector such that sum(M)=1,
        and only S is changed by this function.  In other words, A/sum(A)
        is unchanged by this function.

        The algorithm is a generalized Newton iteration, described in
        \"Estimating a Dirichlet distribution\" by T. Minka.

        Written for Matlab by Tom Minka, ported to Python by Fabian Sinz
        """


        s = sum(a)
        m = a/s

        # sufficient statistics
        if bar_p==None:
            bar_p = mean(log(dat.X),1)

        bar_p = sum(m*bar_p)



        for i in xrange(maxiter):
            old_s = s
            g = digamma(s) - sum(m*digamma(s*m)) + bar_p
            h = trigamma(s) - sum((m**2)*trigamma(s*m));

            success = False
            if not success and (g + s*h) < 0:
                s = 1/(1/s + g/h/s**2.0)
                if s > 0:
                    success = True
                else:
                    s = old_s


            if not success:
                # Newton on log(s)
                s = s*exp(-g/(s*h + g))
                if s > 0:
                    success = True
                else:
                    s = old_s



            if not success:
                # Newton on 1/s
                s = 1/(1/s + g/(s**2.0 *h + 2.0*s*g))
                if s > 0:
                    success = True
                else:
                    s = old_s

            if not success:
                # Newton
                s = s - g/h
                if s > 0:
                    success = True
                else:
                    s = old_s

            if not success:
                raise Errors.UpdateError("All updates failed")
            a = s*m

            if max(abs(s - old_s)) < tol:
                break
        return a

def dirichlet_fit_m(dat, a, bar_p=None, niter=1000,tol=1e-6):
        """
        DIRICHLET_FIT_M   Maximum-likelihood Dirichlet mean.
        
        DIRICHLET_FIT_M(data,a) returns the MLE (a) for the matrix DATA,
        subject to a constraint on sum(A).
        Each row of DATA is a probability vector.
        A is a row vector providing the initial guess for the parameters.
        A is decomposed into S*M, where M is a vector such that sum(M)=1,
        and only M is changed by this function.  In other words, sum(A)
        is unchanged by this function.
        
        The algorithm is a generalized Newton iteration, described in
        \"Estimating a Dirichlet distribution\" by T. Minka.

        Written for Matlab by Tom Minka, ported to Python by Fabian Sinz
        """

        diter = 4

        # sufficient statistics
        if bar_p==None:
            bar_p = mean(log(dat.X),1)


        for i in xrange(niter):
            sa = sum(a)
            old_a = a
            # convergence is guaranteed for any w, but this one is fastest
            w = a/sa
            g = sum(w*(digamma(a)-bar_p)) + bar_p
            a = inv_digamma(g,diter);
            # project back onto the constraint
            a = a/sum(a)*sa
    

            if max(abs(a - old_a)) < tol:
                break
        return a







def dirichletMomentMatch(p):
    """ Each column of p is a multivariate observation on the probability
        simplex. Written for Matlab (fastfit) by Tom Minka, ported to
        Python by Fabian Sinz.
    """
    

    a = mean(p,1)
    m2 = mean(p*p,1)
    ok = where(a > 0)
    s = (a[ok] - m2[ok]) / (m2[ok] - a[ok]**2)
    # each dimension of p gives an independent estimate of s, so take the median.
    s = median(s)
    a = a*s
    return a

if __name__=="__main__":
    p = Dirichlet()
    print p.param['alpha']
    dat = p.sample(10000)
    #    print p.loglik(dat)
    p = Dirichlet()
    print p.param['alpha']
    p.estimate(dat)
    print p.param['alpha']
     

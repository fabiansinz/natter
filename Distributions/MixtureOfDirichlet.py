import Distribution
import Data
from numpy import where, zeros, sum, cumsum, array, size, float64, dot, reshape, log, exp
from numpy import ones, kron, shape, squeeze, Inf, argmax, mean, abs, median, all
from numpy.random import rand
from numpy.random.mtrand import dirichlet
from scipy.special import gammaln, polygamma
from scipy.optimize import fmin_l_bfgs_b
from Auxiliary import Errors
from Auxiliary.Numerics import logsumexp, inv_digamma, digamma, trigamma
import types
import sys
from Dirichlet import dirichlet_fit_m, dirichlet_fit_s, dirichletMomentMatch

class MixtureOfDirichlet(Distribution.Distribution):
    """
      Mixture Of Dirichlet Distributions

      Parameters and their defaults are:

         K:     number of mixtures (default K=3)
         alpha: alpha parameters for the mixture components
                (dimensions X mixtures)  default alpha=rand(3,K)
         pi:    prior mixture probabilities (default normalized
                random array of lenth L)
         
    """

    maxiter = 1000
    innermaxiter = 10
    alltol = 1e-7
    tol = 1e-8

    def __init__(self,param=None):
        self.name = 'Mixture Of Dirichlet Distributions'
        self.param = {'K':3,'n':3}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        if not self.param.has_key('alpha'):
            self.param['alpha'] = rand(self.param['n'],self.param['K'])*3.0
        else:
            self.param['n'] = size(self.param['alpha'],0)
        if not self.param.has_key('pi'):
            self.param['pi'] = rand(self.param['K'])
            self.param['pi']  = self.param['pi']/ sum(self.param['pi'])
        self.primary = ['pi','alpha']
        
        
    def sample(self,m):
        '''

           sample(m)

           samples M examples from the mixture of Dirichlet distributions.
           
        '''
        u = rand(m)
        cum_pi = cumsum(self.param['pi'])
        k = array(m*[0])
        for i in range(m):
            for j in range(len(cum_pi)):
                if u[i] < cum_pi[j]:
                    k[i] = j
                    break
                 
        X = zeros((size(self.param['alpha'],0),m))
        for i in range(self.param['K']):
            I = where(k == i)[0]
            X[:,I] = dirichlet(self.param['alpha'][:,i],len(I)).transpose()
        return Data.Data(X,str(m) + ' samples from a mixture of ' + str(self.param['K']) + ' Dirichlet distributions.')
    
    def loglik(self,dat):
        ret = zeros((self.param['K'],dat.size(1)))
        for k in range(self.param['K']):
            ret[k,:]  = log(self.param['pi'][k]) + squeeze(self.__kloglik(dat,k))
        return logsumexp(ret,0)
#        return log(self.pdf(dat))

    
    def pdf(self,dat):
        ret = zeros((dat.size(1)))
        for k in range(self.param['K']):
            ret  += self.param['pi'][k]*squeeze(exp(self.__kloglik(dat,k)))
        return ret
        #return exp(self.loglik(dat))

    def getPosteriorWeights(self,dat):
        n,m = dat.size()
        K = self.param['K']
        eta = zeros((K,m))
        for k in xrange(K):
            eta[k,:] = log(self.param['pi'][k]) + self.__kloglik(dat,k)
        eta = eta - kron(ones((K,1)),logsumexp(eta,0))
        return exp(eta)

    def estimate(self,dat,method='bfgs',initial_guess=True):
        n,m = dat.size()
        K = self.param['K']
        all_old = Inf
        alpha = self.param['alpha'].copy()
        
        # ------------ get initial guess ------------
        if initial_guess:
            print "\tGetting initial guess by moment matching"
            eta = zeros((K,m))
            pi = self.param['pi']
            for k in xrange(K):
                eta[k,:] = log(pi[k]) + self.__kloglik(dat,k)

            eta = eta - kron(ones((K,1)),logsumexp(eta,0))
            eta = exp(eta)
            etamap = argmax(eta,0)
            for k in xrange(K):
                I = where(etamap == k)[0]
                if len(I) > 0:
                    alpha[:,k] = dirichletMomentMatch(dat.X[:,I])

        #==== start EM itertion ====================================
        print "\tEstimating precision and mean with " + method
        for iteration in xrange(self.maxiter):
            # E-Step: compute the etas
            eta = zeros((K,m))
            pi = self.param['pi']
            for k in range(K):
                eta[k,:] = log(pi[k]) + self.__kloglik(dat,k)
            eta = eta - kron(ones((K,1)),logsumexp(eta,0))
            eta = exp(eta)
            
            changed_method = False
            #======================================================
            # M-Step: maximize w.r.t to alpha and pi
            try:
                if method=='bfgs':
                    alpha = alpha.flatten()
                    bounds = [(0.01,None) for elem in alpha]
                    f = lambda t: self.__objective(t,dat,eta)
                    fprime = lambda t: self.__objective(t,dat,eta,True)
                    alpha = fmin_l_bfgs_b(f,alpha,fprime,bounds=bounds)[0]
                    self.param['alpha'] = reshape(alpha,shape(self.param['alpha']))
                #------------------------------------------------------
                elif method=='kmeans':
                    etamap = argmax(eta,0)
                    for k in xrange(K):
                        I = where(etamap == k)
                        if len(I[0]) > 0:
                            self.__subestimate(dat[:,I[0]],k)
                #------------------------------------------------------
                elif method=='fixpoint':

                    s = sum(alpha,0)
                    w = alpha / kron(ones((n,1)),s)

                    for k in xrange(K):
                        if s[k] <= 0.0:
                            print "\t\tbad initial guess for s[%d]! Fixing it ..." % (k,)
                            if abs(s[k] ) < 10e-10:
                                alpha[:,k] = ones((n,1)) / float(n)
                            else:
                                alpha[:,k] = alpha[:,k] / s[k]
                            s[k] = 1.0
                    for i in xrange(self.innermaxiter):
                        old_alpha = alpha
                        alpha = dirichlet_mix_fit_s(dat, alpha, eta)
                        alpha = dirichlet_mix_fit_m(dat, alpha, eta)
                        if max(abs(alpha - old_alpha).flatten()) < self.tol:
                            break
                    self.param['alpha'] = alpha

                if changed_method:
                    changed_method = False
                    method = old_method
                    print "\tswitching back to fitting method " + str(method) + "..."
            except Errors.UpdateError:
                old_method = method
                method = 'bfgs'
                changed_method = True
                print "\tswitching to fitting method " + str(method) + "..."
            #------------------------------------------------------
            self.param['pi'] = sum(eta,1)/sum(sum(eta))
            #======================================================


            all =  self.all(dat)
            print "\r\tALL at iter No. %d =%.10f           " % (iteration+1,all) ,
            sys.stdout.flush()
            if abs(all-all_old) < self.alltol:
                break
            else:
                all_old = all
        print "\tEM terminated"
            

    def __subestimate(self,dat,k, tol=1e-6):
        
        bar_p = mean(log(dat.X),1)
        a = dirichletMomentMatch(dat.X)
        s = sum(a)
        if s <= 0.0:
            if s == 0.0:
                a = ones(shape(a))/float(len(a))
            else:
                a = a/s
            s = 1.0

        for i in range(self.maxiter):
            old_s = s
            a = dirichlet_fit_s(dat, a, bar_p)
            s = sum(a)
            a = dirichlet_fit_m(dat, a, bar_p, 1) 
            m = a/s
            if abs(s - old_s) < tol:
                break
        self.param['alpha'][:,k] = a
        

    def __objective(self,alpha,dat,eta, df=False):
        alpha = reshape(alpha,shape(self.param['alpha']))
        if not df:
            return -(sum( ( gammaln(sum(alpha,0)) - sum(gammaln(alpha),0)) * sum(eta,1) ) + \
                sum(sum(dot(log(dat.X), eta.transpose())*alpha)))
        else:
            eta_bar = sum(eta,1)
            sum_alpha = sum(alpha,0)
            tmp = dot(log(dat.X), eta.transpose())
            df = zeros(shape(alpha))
            for l in range(self.param['K']):
                df[:,l] = (digamma(sum_alpha[l]) - digamma(alpha[:,l]))*eta_bar[l] + tmp[:,l]
            return -df.flatten()

        
    def __kloglik(self,dat,k):
        alpha = self.param['alpha'][:,k]
        return gammaln(sum(alpha)) - sum(gammaln(alpha)) + dot(reshape(alpha-1.0,(1,dat.size(0))),log(dat.X))


def dirichlet_mix_fit_m(dat, a, eta, niter=1000,tol=1e-8):
    K,m = shape(eta)
    n = dat.size(0)

    diter = 4

    # sufficient statistics
    eta_bar = sum(eta,1)
    xi = dot(log(dat.X),eta.transpose()) / kron(ones((n,1)),eta_bar)

    for i in xrange(niter):
        sa = sum(a,0)
        old_a = a

        w = a / kron(ones((n,1)),sa)

        g = sum(w*(digamma(a)-xi),0) + xi
        a = inv_digamma(g,diter);
        # project back onto the constraint
        a = a * kron(ones((n,1)),sa/sum(a,0))
 
        if max(abs(a - old_a).flatten()) < tol:
            return a
    return a


def dirichlet_mix_fit_s(dat,a,eta,maxiter=100,tol=1e-8):

    K,m = shape(eta)
    n = dat.size(0)
    s = sum(a,0)
    w = a / kron(ones((n,1)),s)

    # get sufficient statistics
    eta_bar = sum(eta,1)
    xi = dot(log(dat.X),eta.transpose()) / kron(ones((n,1)),eta_bar)


    for i in xrange(maxiter):
        failure = ones(K)
        fail_i = where(failure)[0]
        old_s = s
        g = (digamma(s) - sum(w*(digamma(a)-xi),0) ) * eta_bar
        h = (trigamma(s) - sum(w**2.0 *trigamma(a),0) ) *  eta_bar
        # second order approximation
        if any(failure) and any( (g[fail_i] + s[fail_i]*h[fail_i]) < 0.0):
            #print "(1)",
            s[fail_i] = 1.0 / ( 1.0/s[fail_i] + g[fail_i]/ h[fail_i] / s[fail_i]**2.0 )
            s,fail_i,failure = checkForSuccess(s,old_s,fail_i,failure)

        # Newton on log(s)
        if any(failure):
            #print "(2)",
            s[fail_i] = s[fail_i]*exp(-g[fail_i]/(s[fail_i]*h[fail_i] + g[fail_i]))
            s,fail_i,failure = checkForSuccess(s,old_s,fail_i,failure)

        # Newton on 1/s
        if any(failure):
            #print "(3)",
            s[fail_i] = 1.0/(1.0/s[fail_i] + g[fail_i]/(s[fail_i]**2.0 *h[fail_i] + 2.0*s[fail_i]*g[fail_i]))
            s,fail_i,failure = checkForSuccess(s,old_s,fail_i,failure)

        # Newton
        if any(failure):
            #print "(4)",
            s[fail_i] = s[fail_i] - g[fail_i]/h[fail_i]
            s,fail_i,failure = checkForSuccess(s,old_s,fail_i,failure)

        if any(failure):
            raise Errors.UpdateError("All updates failed")

        a = w*kron(ones((n,1)),s)
        if max(abs(s - old_s)) < tol:
            break
    return a

def checkForSuccess(s,old_s,fail_i,failure):
    for ii in xrange(len(fail_i)):
        if s[fail_i[ii]] > 0:
            failure[fail_i[ii]] = 0.0
        else:
            s[fail_i[ii]] = old_s[fail_i[ii]]
    fail_i = where(failure)[0]
    return (s,fail_i,failure)
            

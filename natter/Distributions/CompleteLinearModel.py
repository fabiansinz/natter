from Distribution import Distribution
from natter.DataModule import Data
from natter.Transforms import LinearTransform
from numpy import Inf, array, real, max, arccos, diag, dot, pi, mean, abs, diff, sum, log
from natter.Auxiliary.Optimization import StGradient
from mdp.utils import random_rot

class CompleteLinearModel(Distribution):
    '''
      COMPLETE LINEAR MODEL

      implements a distribution of the form p(x) = q(W*x), where W is
      an orthogonal matrix (i.e. the data has been whitened before).

      IMPORTANT: W is assumed to be an element of the special
      orthogonal group, i.e. the determinant of W must be one or minus
      one. If W is only in the general linear group the returned
      likelihoods might not be correct.

      Parameters and their defaults are:

         q:  base distribution (default q=None)

         W: linear filter object (default is either a random orthogonal
            matrix if q.param[\'n\'] exists or None)
    '''

    def __init__(self,param = None ):
        if param == None:
            param = {'q':None, 'W':None}
        self.name = 'Complete Linear Model'
        self.param = param
        if not param.has_key('W') or \
               not isinstance(param['W'],LinearTransform) and \
               isinstance(param['q'],Distribution) and \
               param['q'].param.has_key('n'):
            self.param['W'] = LinearTransform(random_rot(param['q'].param['n']),\
                                                      'Random rotation matrix',['sampled from Haar distribution'])

        self.primary = ['q','W']
            
    def loglik(self,dat):
        # the determinant of W is not part here since W is in SO(n)
        return self.param['q'].loglik(self.param['W']*dat) 


    def estimate(self,dat,param0=None):
        W = self.param['W'].W
        q = self.param['q']

        # initialization of default optimizatiion parameters
        param = {'tolF':5.0*1e-6, 'maxiter':50, 'SOmaxiter':10, 'searchrange':10,'lsTol':1e-2}
        if param0 != None:
            for k in param0.keys():
                param[k] = param0[k]

        dAngle = array([Inf for dummy in range(param['maxiter']+2)])
        fRec = array([Inf for dummy in range(param['maxiter']+2)])

        iter = 0
        loop = True
        while loop:
            print "\t"+40*'='
            print "\tIteration %d" % (iter+1,)
            
            # estimating the base distribution
            print "\tEstimating Base Distribution ..."
            if param.has_key('args'):
                q.estimate(LinearTransform(W)*dat,*param['args'])
            else:
                q.estimate(LinearTransform(W)*dat)
                
            self.param['q'] = q
            print "\t[Done]"
            print '\t' + q.__str__().replace('\n','\n\t')
            print "\tALL: %.4f [Bits/Component]" %( self.all(dat),)

            # estimating the filter matrix
            (Wnew,fval,param) = StGradient(self.objective, W, \
                                                                   param, dat,q)
            dAngle[iter] = real(max(arccos(diag(dot(W,Wnew.transpose()))) \
                                              /2/pi*360.0))
            fRec[iter] = fval[0]
            W = Wnew
            # check stopping criterion
            if iter > param['maxiter'] or \
                    (iter > 1 and mean(abs(diff(fRec[iter-2:iter+1]))) < 1e-4 and dAngle[iter] < 1):
                print "\t Optimization terminated! [Exiting]"
                loop = False
            iter +=1

            self.param['W'].W = W

            if param.has_key('args'):
                q.estimate(self.param['W']*dat,*param['args'])
            else:
                q.estimate(self.param['W']*dat)
            
        self.param['q'] = q
        
    def objective(self, W, nargout,dat,q):
        """
        The objective function to be optimized with
        Auxiliary.Optimization.StGradient. It computes the mean likelhood
        """
        (n,m) = dat.size()
        if nargout == 1:
            return (sum(q.loglik(Data(array(dot(W,dat.X)))))/m/n/log(2),)
        else:
            return (sum(q.loglik(Data(array(dot(W,dat.X)))))/m/n/log(2), \
                        dot(q.dldx(Data(array(dot(W,dat.X)))),\
                                   dat.X.transpose())/m/n/log(2))

        

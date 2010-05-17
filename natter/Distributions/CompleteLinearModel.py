from Distribution import Distribution
from natter.DataModule import Data
from natter.Transforms import LinearTransform
from numpy import Inf, array, real, max, arccos, diag, dot, pi, mean, abs, diff, sum, log
from natter.Auxiliary.Optimization import StGradient
from mdp.utils import random_rot

class CompleteLinearModel(Distribution):

    def __init__(self,param = None ):
        """
        CompleteLinearModel constructor.
        
        :param param: Initial parameters for the CompleteLinearModel distribution. The CompleteLinearModel distribution has parameters *W* (LinearTransform) and 'q' (the base distribution). The default values for *W* is a random rotation matrix. *q* does not have a default value.
        
        Primary parameters are ['q','W'].
        
        :type param: dict.
        :returns:  A CompleteLinearModel distribution object initialized with the parameters in param.
    
        """
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
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''

        return self.param['q'].loglik(self.param['W']*dat) 


    def estimate(self,dat,param0=None):
        """

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on
        the Distributions module <tutorial_Distributions>`).

        Fitting is carried out by optimizing q on W*x w.r.t to W and
        the parameters of q.

        The optimization can be controlled by setting *param0*. The following parameters can be set:
        
        * '*tolF*' stopping tolerance in the function values
        * '*maxiter*' maximal number of iteration steps
        * '*SOmaxiter*' maximal number of gradient steps on SO(n) in each iteration
        * '*searchrange*' searchrange for the linesearch in the gradient ascent on SO(n)
        * '*lsTol*' linesearch tolerance
        * '*args*' additional arguments for on q.estimate.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param param0: optimization parameters
        :type param0:  dict
        """
        
        W = self.param['W'].W
        q = self.param['q']

        # initialization of default optimizatiion parameters
        param = {'tolF':5.0*1e-6, 'maxiter':50, 'SOmaxiter':10, 'searchrange':10,'lsTol':1e-2}
        if param0 != None:
            for k in param0.keys():
                param[k] = param0[k]

        if 'W' in self.primary:
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

                if 'q' in self.primary:
                    if param.has_key('args'):
                        q.estimate(self.param['W']*dat,*param['args'])
                    else:
                        q.estimate(self.param['W']*dat)
                    self.param['q'] = q
        elif 'q' in self.primary:
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

        

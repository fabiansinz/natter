from Distribution import Distribution
from Gamma import Gamma
from natter.DataModule import Data
from numpy import log, zeros, array, Inf, any, isinf, max, abs, squeeze, sign
from numpy.random import beta, dirichlet, rand
from natter.Auxiliary import LpNestedFunction
from natter.Auxiliary.Optimization import fminboundnD, goldenMinSearch
import sys



class LpNestedSymmetric(Distribution):
    """
    Lp-nested symmetric Distribution

    :param param:
        dictionary which might containt parameters for the Lp-nested symmetric distribution
              'n'    :   Dimensionality (default = 2)
        
              'rp'   :    Radial density (default Gamma())
              
              'f'    :    Lp-nested function (default = LpNestedFunction('(0,0,(1,1:2))',[.5,1.0]))
              
    :type param: dict

    Primary parameters are ['rp','f'].
        
    """

    def __init__(self,param=None):
        self.name = 'Lp-Nested Symmetric Distribution'
        self.param = {'n':2, 'rp':Gamma(),'f':LpNestedFunction('(0,0,(1,1:2))',[.5,1.0])}

        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        self.param['n'] = self.param['f'].n[()]
        self.primary = ['rp','f']

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat. 

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype: numpy.array
           
        '''
        r = self.param['f'].f(dat)
        return self.param['rp'].loglik(r) \
               - self.param['f'].logSurface() - (self.param['n']-1)*log(r.X)

    def sample(self,m):
        """

        Samples m samples from the current LpNestedSymmetric distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """
        ret = zeros((self.param['f'].n[()],m))
        r = beta(float(self.param['f'].n[()]),1.0,(1,m))
        recsample((),r,self.param['f'],m,ret)
        
        ret = Data(ret,'Samples from ' + self.name)
        ret.scale(self.param['rp'].sample(m).X/self.param['f'].f(ret).X)

        return ret

    def dldx(self,dat):
        """

        Returns the derivative of the log-likelihood of the Gamma distribution w.r.t. the data in dat. 
        
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the derivatives.
        :rtype:    numpy.array
        
        """        
        drdx = self.param['f'].dfdx(dat)
        r = self.param['f'].f(dat)
        tmp = (self.param['rp'].dldx(r) - (self.param['n']-1.0)*1.0/r.X)
        for k in range(len(drdx)):
            drdx[k] *= tmp
        return drdx


    def estimate(self,dat,method="neldermead"):
        """

        Estimates the parameters from the data in dat. It is possible
        to only selectively fit parameters of the distribution by
        setting the primary array accordingly (see :doc:`Tutorial on
        the Distributions module <tutorial_Distributions>`).

        Fitting is carried out by alternating between optimizing *f*,
        keeping the parameters of the radial distribution fixed, and
        optimizing the parameters of the radial distribution keeping
        the value of *f* fixed.

        :param dat: Data points on which the Gamma distribution will be estimated.
        :type dat: natter.DataModule.Data
        :param method: method to be used for fitting *f* (default: 'neldermead'; other options are 'greedy')
        :type method:  string
        """


        if 'f' in self.primary:
            # --------- estimation with Nelder-Mead method -------------
            if method == "neldermead":
                print "\tEstimating best p with bounded Nelder-Mead ..."
                f = lambda t: self.__all(t,dat)
                bestp = fminboundnD(f,self.param['f'].p,self.param['f'].lb,self.param['f'].ub,1e-5)
                self.param['f'].p = bestp

            # --------- estimation with greedy method -------------
            elif method == "greedy":
                pind = [] # will store the indices in the order in which they will be estimated later

                # check whether greedy can be used
                ip = self.param['f'].ipdict
                tmp = -1
                for i in ip.keys():
                    if len(ip[i][0]) > tmp:
                       pind.insert(0,i)
                    else:
                        pind.append(i)
                    tmp = len(ip[i][0])
                    for j in range(len(ip[i])):
                        if len(ip[i][j]) != tmp:
                            print "\tGreedy estimation method not applicable here! Falling back to \"neldermead\""
                            self.estimate(dat,"neldermead")
                            return
                # setup iteration
                pold = array(self.param['f'].p)+Inf
                maxiter =10
                itercount = 0
                tol = 1e-2
                UB = array(self.param['f'].ub)
                LB = array(self.param['f'].lb)
                if any(isinf(UB)):
                    print "\tGreedy only works with finite upper and lower bounds! Setting UB to 10*(LB+1.0)"
                    UB = (array(self.param['f'].lb)+1.0) * 10.0

                # iterate
                while itercount < maxiter and max(abs(pold-self.param['f'].p)) > tol:
                    print "\tGreedy sweep No. %d" % (itercount,)
                    pold = array(self.param['f'].p)
                    
                    for i in pind:
                        f = lambda t: self.__all2(t,i,dat)
                        tmp = goldenMinSearch(f,LB[i],UB[i],1e-3)
                        self.param['f'].p[i] = .5*(tmp[0]+tmp[1])
                        print ""
                    if itercount > 0:
                        print "\t... adapting search ranges for p"
                        LB = array(self.param['f'].p)*.8
                        UB = array(self.param['f'].p)*1.2
                    itercount += 1
            
        if 'rp' in self.primary:
            self.param['rp'].estimate(self.param['f'].f(dat))
        print "\t[Done]"

    def __all(self,p,dat):
        if len(p) <= 5:
            print "\r\t" + str(p),
        else:
            print "\r\t[" + str(p[0]) + ", ..., " + str(p[-1]),
        sys.stdout.flush()
        self.param['f'].p = p
        if 'rp' in self.primary:
            self.param['rp'].estimate(self.param['f'].f(dat))
        return self.all(dat)

    def __all2(self,p,i,dat):
        self.param['f'].p[i] = p
        if len(self.param['f'].p) <= 5:
            print "\r\t" + str(self.param['f'].p),
        else:
            print "\r\tp[%d]=%.6f" % (i,p),

        sys.stdout.flush()
        if 'rp' in self.primary:
            self.param['rp'].estimate(self.param['f'].f(dat))
        return self.all(dat)

def recsample(key,r,L,m,ret):
    alpha = array([float(L.n[key + (i,)])/L.p[L.pdict[key]] for i in range(L.l[key])])
    p = L.p[L.pdict[key]]

    tmp = squeeze(array([r*elem**(1/p) for elem in dirichlet(alpha,m).transpose()]))
    for i in range(L.l[key]):
        I = key + (i,)
        if L.n[I] > 1:
            recsample(I,tmp[i,:],L,m,ret)
        else:
            ret[L.i(I),:] = tmp[i,:]*sign(rand(1,m)-.5)
        
def sortMultiIndices(x,y):
    for k in range(min(len(x),len(y))):
        if x[k] < y[k]:
            return -1
        if x[k] > y[k]:
            return 1
    if len(x) < len(y):
        return -1
    if len(y) < len(x):
        return 1
    return 0

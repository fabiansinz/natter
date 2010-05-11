from __future__ import division
from Auxiliary import Potential
from Auxiliary.Optimization import StGradient
from Distribution import Distribution 
from numpy import ones, size, max, zeros, sum, array, kron, reshape,dot,shape,squeeze, Inf
from numpy.random import randn
from Transforms import TransformFactory, LinearTransform
from DataModule import Data
from sys import stdout
from scipy.optimize import fmin_l_bfgs_b


class ProductOfExperts(Distribution):
    '''
      PRODUCT OF EXPERTS

      implements a distribution of the form p(x) = prod q_i(w_i*x)^alpha_i, where W is
      an tall matrix.

      Parameters and their defaults are:

         potentials: List of 1D potentials (Auxiliary.Potential objects; default is a list of Potential(\'laplacian\') objects)
         n:   Dimensionality of the input data (default n=2).
         N:   Number of Filters
         W: linear filter object (Default: 2n X n Linear Filter object)
         alpha: exponents ( ones((2*n)))
    '''

    maxiter = 100
    Tol = 1e-4

    def __init__(self,param0 = None ):
        self.param = {'potentials':None, 'W':None, 'n':2.0,'N':4}
        if param0 != None:
            for k in param0.keys():
                self.param[k] = param0[k]
        self.name = 'Product of Experts'
        if param0==None or not param0.has_key('N'):
            self.param['N'] = 2*self.param['n']
        if param0==None or not param0.has_key('W') or not isinstance(param0['W'],LinearTransform):
            self.param['W'] = TransformFactory.stRND( (self.param['N'],self.param['n']) )
        if param0 == None or not param0.has_key('alpha'):
            self.param['alpha'] = ones((self.param['N'],))
        if param0 == None or not param0.has_key('potentials'):
            self.param['potentials'] = [Potential('studentt') for i in xrange(self.param['N'])]
        self.primary = ['potentials','W','alpha']

    def estimate(self,dat):

        print "\tEstimating parameters of Product of Experts...\n"
        stdout.flush()
        
        alpha = self.param['alpha'].copy()
        W = array(self.param['W'].W.copy())

        estimate_alpha = 'alpha' in self.primary
        estimate_W = 'W' in self.primary

        W_old = Inf*W
        alpha_old = Inf*alpha
        for iteration in xrange(self.maxiter):
            
            if estimate_W:
                print "\testimating W ..."; stdout.flush()
                wobjective = lambda Wt, derivative: ( -self.myscore(alpha,Wt.transpose(),dat.X,"score"),) if derivative == 1 \
                             else  ( -self.myscore(alpha,Wt.transpose(),dat.X,"score"), -self.myscore(alpha,Wt.transpose(),dat.X,"dW").transpose())
                (W,fval,param) = StGradient(wobjective, W.transpose())
                W = W.transpose()

            if estimate_alpha:
                print "\testimating alpha ..."; stdout.flush()
                af = lambda a: self.myscore(a,W,dat.X,"score")
                afprime = lambda a: self.myscore(a,W,dat.X,"dalpha")
                bounds = [(1e-12, Inf) for dummy in xrange(self.param['N'])]
                alpha = fmin_l_bfgs_b(af,alpha,afprime,bounds=bounds)[0]

            if max(abs(W-W_old).flatten()) < self.Tol and max(abs(alpha-alpha_old)) < self.Tol:
                print "\t... optimization terminated!"
                self.param['alpha'] = alpha
                self.param['W'].W = W
                break
            else:
                W_old= W.copy()
                alpha_old = alpha.copy()

                
    def myscore(self,alpha,W,X,otype):
        Y = Data(dot(W,X))
        n,m = shape(X); m = float(m); 
        N = self.param['N']
        pot = self.param['potentials']

        D1 = zeros((N,m))
        D2 = zeros((N,m))
        for k in xrange(N):
            D1[k,:] = pot[k].dlogdx(Y[k,:])
            D2[k,:] = pot[k].d2logdx2(Y[k,:])

        
        if otype == "score":
            return 1/m*sum(alpha*sum(D2,1)*sum(W**2,1)) \
                   + 1.0/2.0/m * sum(sum(dot(D1.transpose(),kron(reshape(alpha,(N,1)),ones((1,n)))*W)**2.0))

        if otype == "dalpha":
            return 1.0/m*sum(D2,1)*sum(W**2,1) \
                    + 1.0/m* sum( dot(D1, dot(D1.transpose(),kron(reshape(alpha,(N,1)),ones((1,n)))*W))* W,1)
                
    
        if otype == "dW":
            D3 = zeros((N,m))
            for k in xrange(len(pot)):
                D3[k,:] = pot[k].d3logdx3(Y[k,:])

            tmp2 = dot(D1.transpose(),kron(reshape(alpha,(N,1)),ones((1,n)))*W) # tmp2 is m X n


            tmp = 1/m * dot(D3,X.transpose()) * kron(  reshape(alpha*sum(W**2.0,1),(N,1))  ,  ones((1,n))  )
            tmp += 2/m * W *kron(  reshape(alpha*sum(D2,1),(N,1))  ,  ones((1,n))  )

            tmp += 1/m * dot(D1,tmp2) * kron(  reshape(alpha,(N,1))  ,  ones((1,n))  )
             
            
            tmp2 = dot(tmp2,W.transpose())

            for p in xrange(N):
                for q in xrange(n):
                    tmp[p,q] = tmp[p,q] + 1/m * alpha[p] * sum(D2[p,:]*X[q,:]*tmp2[:,p])
            return tmp
            

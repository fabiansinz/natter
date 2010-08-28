from __future__ import division
from sys import stdout 
from numpy import shape, isfinite, abs, pi, arcsin, reshape, zeros, Inf, max, dot, real,sin,  min,array
#from numpy.linalg import svd
from scipy.linalg import sqrtm, inv
from scipy.optimize import fminbound
from scipy.optimize import fmin
from natter.Auxiliary import Errors

def fminboundnD(f,x0,LB,UB,tol=1e-3,*args):
    """
    I ported this function from Matlab. Someone else wrote it. TODO:
    Who was it and reference him.
    """
    xsize = shape(x0);
    x0 = x0.flatten()
    n=len(x0);
    
    if (n!=len(LB)) or (n!=len(UB)):
        raise Errors.DimensionalityError('x0 is incompatible in size with either LB or UB.')


    # 0 --> unconstrained variable
    # 1 --> lower bound only
    # 2 --> upper bound only
    # 3 --> dual finite bounds
    # 4 --> fixed variable
    BoundClass = [0]*n

    for i in xrange(n):
        k = isfinite(LB[i]) + 2*isfinite(UB[i])
        BoundClass[i] = k;
        if (k==3) and (LB[i]==UB[i]):
            BoundClass[i] = 4;

    # transform starting values into their unconstrained
    # surrogates. Check for infeasible starting guesses.
    x0u = x0.copy()
    k=0

    for i in xrange(n):
        if BoundClass[i] == 1:
            # lower bound only
            if x0[i]<=LB[i]:
                # infeasible starting value. Use bound.
                x0u[k] = 0.0
            else:
                x0u[k] = abs(x0[i] - LB[i])
            k+=1
        elif BoundClass[i] == 2:
            # upper bound only
            if x0[i]>=UB[i]:
                # infeasible starting value. use bound.
                x0u[k] = 0.0
            else:
                x0u[k] = abs(UB[i] - x0[i])
            k+=1
        elif BoundClass[i] == 3:
            # lower and upper bounds
            if x0[i]<=LB[i]:
                # infeasible starting value
                x0u[k] = -pi/2.0
            elif x0[i]>=UB[i]:
                # infeasible starting value
                x0u[k] = pi/2
            else:
                x0u[k] = 2*(x0[i] - LB[i])/(UB[i]-LB[i]) - 1.0
                x0u[k] = 2.0*pi+arcsin(max(array([-1.0,min(array([1.0,x0u[k]]))])))
            k+=1
        elif BoundClass[i] == 0:
            x0u[k] = x0[i]
            k+=1

            
    if k<=n:
        x0u = x0u[:k]

    # were all the variables fixed?
    if len(x0u) == 0:
        # All variables were fixed. quit immediately, setting the
        # appropriate parameters, then return.
  
        # undo the variable transformations into the original space
        x = xtransform(x0u,LB,UB,BoundClass,n)
  
        # final reshape
        x = reshape(x,xsize);
        return x


    # now we can call fmin
    f2 = lambda t: f(xtransform(t,LB,UB,BoundClass,n),*args)

    xu = fmin(f2,x0u,xtol=tol,*args)
    # undo the variable transformations into the original space
    x = xtransform(xu,LB,UB,BoundClass,n)
    
    # final reshape
    return reshape(x,xsize);

def xtransform(x,LB,UB,BoundClass,n):
    # converts unconstrained variables into their original domains

    xtrans = zeros((n,))
    k=0
    for i in xrange(n):
        if BoundClass[i] == 1:
            # lower bound only
            xtrans[i] = LB[i] + abs(x[k])
            k+=1
        elif BoundClass[i] == 2:
            # upper bound only
            xtrans[i] = UB[i] - abs(x[k])
            k+=1
        elif BoundClass[i] == 3:
            # lower and upper bounds
            xtrans[i] = (sin(x[k])+1.0)/2.0
            xtrans[i] = xtrans[i]*(UB[i] - LB[i]) + LB[i]
            # just in case of any floating point problems
            xtrans[i] = max(array([LB[i],min(array([UB[i],xtrans[i]]))]))
            k+=1
        elif BoundClass[i] == 4:
            # fixed variable, bounds are equal, set it at either bound
            xtrans[i] = LB[i]
        elif BoundClass[i] == 0:
            xtrans[i] = x[k];
            k+=1
    return xtrans

def goldenMinSearch(f, xi, xf, t=1.0e-9, verbose=False):
    """
    (XI,XF,NOITER) = GOLDENMINSEARCH(FUNC, XI, XF, TOL)
    
    performs a golden search to find the minimum of FUNC. which takes
    a single real number, in the interval [XI,XF]. It returns the
    interval borders XI and XF after they have been narrowed down, as
    well as the number iterations it took the find them. TOL specifies
    the stopping criterion. 
    """
    # constants
    A = 0.6180339887
    
    n = 0
    
    x1 = xi
    x2 = xf

    x3 = x1 + A * (x2-x1)
    x4 = x2 - A * (x2-x1)
    
    fx3 = f(x3)
    fx4 = f(x4)
    
    while abs(x2-x1) > t:
        #print n, x1, x2, abs(x2-x1)
        n+=1
        if fx3 < fx4:
        # fx4 > fx3
            x1 = x4
            x4 = x3
            x3 = x1 + A * (x2-x1)
        else:
         # fx4 < fx3
            x2 = x3
            x3 = x4
            x4 = x2 - A * (x2-x1)
        
        fx3 = f(x3)
        fx4 = f(x4)
        if verbose:
            print "\r\tGolden Search (%.4g, %.4g, %d)" % (x1,x2,n),
        stdout.flush()
    if verbose:
        print '\n'
    return (x1,x2,n)

def goldenMaxSearch(f, xi, xf, t=1.0e-9,verbose=False):
    """
    (XI,XF,NOITER) = GOLDENMAXSEARCH(FUNC, XI, XF, TOL)
    
    performs a golden search to find the maximum of FUNC. which takes
    a single real number, in the interval [XI,XF]. It returns the
    interval borders XI and XF after they have been narrowed down, as
    well as the number iterations it took the find them. TOL specifies
    the stopping criterion. 
    """
    g = lambda x: -1.0*f(x)
    return goldenMinSearch(g, xi,xf,t,verbose)

def StGradient(func, X, param0=None, *args):
    """
    (X,FVAL,PARAM) = StGRADIENT(FUNC, X, PARAM, *ARGS)
    
    performs gradient ascent on the special orthogonal group as described in 
   
    J. Manton, \"Optimization algorithms exploiting unitary
    constraints,\" Signal Processing, IEEE Transactions on, vol. 50,
    2002, pp. 635-650.

    FUNC is a function that it called in the following way:
    (VAL, DF) = FUNC(X, 2, *ARGS)
    (DF,)     = FUNC(X, 1, *ARGS)
      
    i.e. the second argument is either 2 in which case it returns the
    function value at X in SO(n), or the second argument equals one in
    which case it only returns the function value at X. 

    The dictionary PARAM lets you set optimization parameters like the
    termination threshold on the supremums norm of the gradient
    (tolF), the maximum number of iterations (SOmaxiter) or the
    searchrange for the linesearch (searchrange). The default values for PARAM are

    PARAM = {'tolF':1e-8, 'SOmaxiter':20, 'searchrange':10}

    *ARGS are arguments that are passed down to FUNC (see above).
    """
    
    bestdelta = Inf
    Z = Inf
    param = {'tolF':1e-8, 'SOmaxiter':20, 'searchrange':10.0,'lsTol':1e-6,'linesearch':'golden'}
    if param0 != None:
        for k in param0.keys():
            param[k] = param0[k]
    tolF = param['tolF']
    maxiter = param['SOmaxiter']
    b = float(param['searchrange'])
    k = 1
    while max(max(abs(Z))) > tolF:
        ( ftmax, Df) = func(X,2,*args)
        print int(k==1)*"\t" + "fval=%.6f..." % (ftmax,),
        Z = Df - dot(dot(X,Df.transpose()),X)
        print "\t[Done]"

        print "\tLinesearch %i" % (k,),
        F = lambda t: func(projectOntoSt(X+t*Z),1,*args)[0]
        if param['linesearch'] == 'golden':
            bestdelta = goldenMaxSearch(F,0,b,param['lsTol'])
            bestdelta = .5*(bestdelta[0]+bestdelta[1])
        elif param['linesearch'] == 'brent':
            bestdelta = fminbound(F,0,b,(),param['lsTol'])
            bestdelta = bestdelta[0]

        if bestdelta < b/4.0:
            b /=2
            print "-(%.4f) " % b,
        elif bestdelta < 3.0*b/4.0:
            b *=2
            print "+(%.4f) " % b,
        else:
            print ".(%.4f) " % b,
  
        X = projectOntoSt(X+bestdelta*Z)
        ftmax = func(X,1,*args)
        k += 1;
        if k > maxiter:
             print 'Maxiter reached! Exiting ...\n'
             break
    param['b'] = b
    if k <= maxiter:
        print "\tMaximal gradient entry is smaller than %.4g! Exiting ...\n" % (tolF,)
    return (X,ftmax,param)

def projectOntoSt(C):
    # (u,d,v) = svd(C,full_matrices=False)
    # return dot(u,v)
    return real( dot(  inv(sqrtm(dot(C,C.T)))  ,C) )
    
def checkGrad(f,x,tol,*args):
    """
    BOOL = CHECKGRAD(FUNC, X, TOL, *ARGS)

    checks if the gradient returned by FUNC if correct. FUNC is called
    in the following way

    FUNC is a function that it called in the following way:
        (VAL, DF) = FUNC(X, 2, *ARGS)
        (VAL,)     = FUNC(X, 1, *ARGS)

    i.e. the second argument is either 2 in which case it returns the
    function value at X in SO(n), or the second argument equals one in
    which case it only returns the function value at X. 

    TOL specifies the tolerance threshold on the supremum norm of the
    difference between the gradients.

    *ARGS are passed down to FUNC.
    """
    h = 1e-6
    df2 = Inf*x
    (fval,df) = f(x,2,*args)
    sh = shape(x)
    if len(sh) == 2:
        for i in xrange(sh[0]):
            for j in xrange(sh[1]):
                y = x.copy()
                y[i,j] += h
                df2[i,j] = (f(y,1,*args)[0]-f(x,1,*args)[0])/h
    if len(sh) == 1:
        for i in xrange(sh[0]):
            y = x.copy()
            y[i] += h
            df2[i] = (f(y,1,*args)-f(x,1,*args))/h
    return max(abs(df-df2).flatten()) < tol


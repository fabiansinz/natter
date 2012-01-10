from natter.Distributions import Gaussian
from scipy.optimize import fmin_bfgs, fmin_ncg
from numpy import zeros
from natter.DataModule import Data
from numpy.linalg import cholesky

def laplaceApproximation(listOfDist,initPoint=None):
    """
    Function that calculates the Laplace approximation for the given
    list of Distributions. It returns a Gaussian Distribution with the
    mean set to the mode of the product of the Distributions given in
    the input list and the variance to the negative, inverse Hessian
    of that product.

    TODO: we probably dont want to use distributions but functions here.

    The distributions in the list are assumed to have implemented dldx
    as well as dldx2 method (first and second derivative)

    :param listOfDist: List of distributions from which the product is build.
    :type listOfDist: list of natter.Distributions
    
    """

    # chech argument
    for dist in listOfDist:
        OK = hasattr(dist,'dldx')
        OK = OK and hasattr(dist,'dldx2')
        if not OK:
            raise ValueError('Distribution has not implemented dldx or dldx2')
    if initPoint is None:
        initPoint = listOfDist[0].sample(1).X.flatten()

    def f(x):
        fx = 0.0
        for dist in listOfDist:
            fx -= dist.loglik(x)
        return fx
    def df(x):
        grad = zeros(len(initPoint.flatten()))
        for dist in listOfDist:
            grad = grad - dist.dldx(Data(x))
        return grad
    def ddf(x):
        H = zeros((len(initPoint.flatten()),len(initPoint.flatten())))
        for dist in listOfDist:
            H = H - dist.dldx2(Data(x))
        return H

    xmin, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = \
          fmin_bfgs(f,initPoint,fprime=df)
    Hopt = ddf(xmin)
    Lh = cholesky(Hopt)
    laplaceApprox = Gaussian({'n':len(xmin.flatten())})
    laplaceApprox['mu'] = xmin
            
    

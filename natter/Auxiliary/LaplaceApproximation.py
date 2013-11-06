from natter.Distributions import Gaussian
from scipy.optimize import fmin_bfgs
from numpy import zeros
from natter.DataModule import Data
from numpy.linalg import inv

def laplaceApproximation(listOfDist,initPoint=None):
    """Function that calculates the Laplace approximation for the product
    of likelihood functions associated with the the given list of
    Distributions. It returns a Gaussian Distribution with the mean
    set to the mode of the product of the likelihoods given in the
    input list and the variance to the negative, inverse Hessian of
    that product. Note that the resulting distribution is a
    distribution over parameters, whereas the input list is a
    distribution over datapoints. Therefore, to include a prior
    distribution over parameters within the product using this
    implementation, a particularly designed data-distribution has to
    be implemented.


    The distributions in the list are assumed to have implemented dldx
    as well as dldx2 method (first and second derivative)

    :param listOfDist: List of distributions from which the product is build as a product over the corresponding likelihood functions.
    :type listOfDist: list of natter.Distributions
    :param initPoint: Initial point for the mean of the laplace Approximation (optional argument). If None is given, then the parameters from the first distribution in the list are taken as an initial point.
    :type initPoint: numpy.ndarray
    :returns: laplace approximated distribution
    :rtype: natter.Distributions.Gaussian

    """

    # check argument
    for dist in listOfDist:
        OK = hasattr(dist,'dldx')
        OK = OK and hasattr(dist,'dldx2')
        if not OK:
            raise ValueError('Distribution has not implemented dldx or dldx2')
    if initPoint is None:
        initPoint = listOfDist[0].primary2array()#sample(1).X.flatten()

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
    laplaceApprox = Gaussian({'n':len(xmin.flatten())})
    laplaceApprox['mu'] = xmin
    laplaceApprox['sigma'] = inv(Hopt)
    return laplaceApprox


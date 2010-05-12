from __future__ import division
from natter.Distribution import Gaussian
from scipy import optimize
from numpy import sum,log,exp




def noiseContrastive(modelDistribution,
                     data,
                     noiseDistribution=None,
                     verbosity=-1):
    """
    Implementation of the noise-contrastive estimation algorithm by Hutmann & Hyvaerinen.

    If no noise distribution is given a standard Gaussian is used as a noise contrastive.
    
    References:
    Gutmann, M. and Hyvaerinen, A. (2010) Noise-contrastive
    estimation: A new estimation principle for unnormalized
    statistical models. (AISTATS)
    """
    
    dim,nsamples = data.size()
    if noiseDistribution==None:
        d ={'n':dim}
        noiseDistribution = Gaussian(d)
    dataNoise = noiseDistribution.sample(nsamples);
    

    def f(theta):
        modelDistribution.array2primary(theta)
        return noiseContrastiveObjective(modelDistribution,noiseDistribution,data,dataNoise)
    def df(theta):
        modelDistribution.array2primary(theta)
        return gradNoiseContrastiveObjective(modelDistribution,noiseDistribution,data,dataNoise)

    theta0 = modelDistribution.primary2array()
    if verbosity>0:
        thetaOpt = optimize.fmin_bfgs(f,theta0,df,disp=1)
    else:
        thetaOpt = optimize.fmin_bfgs(f,theta0,df,disp=0)
    



def noiseContrastiveObjective(modelDistribution,noiseDistribution,dataModel,dataNoise):
    """
    Objective function for the noise contrastive, basically
    classification performance of the current modelDistribution on
    classifying samples into data and noise. Specifically:

    J(\theta) = \sum -ln(1+ noisePDF/modelPDF (X)) + ln( noisePDF/(noisePDF+modelPDF)(Y))
    where X data and Y noise.
    
    
    """

    hx =  logistic(modelDistribution.loglik(dataModel) -noiseDistribution.loglik(dataModel))
    hy =  logistic(modelDistribution.loglik(dataNoise) -noiseDistribution.loglik(dataNoise))
    J = -1/(2*dataModel.size(0))*sum(log(hx) + log(1-hy))
    return J


def gradNoiseContrastiveObjective(modelDistribution,noiseDistribution,dataModel,dataNoise):
    """
    Returns the gradient of the objective with respect to the primary
    parameters of the model distribution.
    
    Arguments:
    - `modelDistribution`: 
    - `noiseDistribution`:
    - `dataModel`:
    - `dataNoise`:
    """
    hx = logistic(modelDistribution.loglik(dataModel) -noiseDistribution.loglik(dataModel))
    hy = logistic(modelDistribution.loglik(dataNoise) -noiseDistribution.loglik(dataNoise))
    print "Size hx: " + str(hx.shape)
    print "size dldtheta " + str(modelDistribution.dldtheta(dataModel).shape)
    g1 = sum((1-hx)*modelDistribution.dldtheta(dataModel),axis=1)
    g2 = sum((-hy)*modelDistribution.dldtheta(dataNoise),axis=1)
    return -(g1+g2)



def logistic(x):
    """
    Returns the logistic function:

    1/(1+exp(-x))
    """
    return 1/(1+ exp(x))


    

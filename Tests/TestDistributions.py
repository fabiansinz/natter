from __future__ import division
from numpy import log,pi,sum,ones,sqrt,logspace
from numpy.random import randn
from numpy.linalg import norm
import unittest
from scipy.optimize import check_grad, approx_fprime
import numpy as np
from config_dictionaries import distributions_to_test
from natter.Distributions import *
from natter.Auxiliary.Numerics import logsumexp
from copy import deepcopy as cp
from natter.Auxiliary.Errors import AbstractError
from natter.Distributions import Gaussian

def test_loglik():
    """
    Testing the log-likelihood methods for all given distributions.
    """
    for dic in distributions_to_test:
        yield check_loglik_importance,dic

def test_sample():
    """
    Testing the sampling methods for all given distributions.
    """
    for dic in distributions_to_test:
        yield check_sample, dic

def test_array2primary():
    """
    Testing if the primary parameters can be set by an array and vice versa, that is can be read out.
    """
    for dic in distributions_to_test:
        yield check_array2primary, dic


def test_dldtheta():
    """
    Testing the gradient wrt to the primary parameters, if such a function is provided by the distributions.
    """
    for dic in distributions_to_test:
        yield check_dldtheta, dic

def test_dldx():
    """
    Testing the gradient wrt to the data, if such a function is provided by the distributions.
    """
    for dic in distributions_to_test:
        yield check_dldx,dic

def test_default():
    """
    Testing, if it is possible to generate a default distributions, i.e. a call without any arguments lead to a proper distribution.
    """
    for dic in distributions_to_test:
        yield check_default,dic

def test_basic():
    """
    Check, if basic methods are implemented, right now, sample and loglik are tested for.
    """
    for dic in distributions_to_test:
        yield check_basic,dic

        
def fill_dict_with_defaults(dic):
    """
    Helper function, which sets missing values for a given dictionary containing information about the distribution to test.

    Argument:
    :param dic: dictionary containing information about the distribution to test.
    :type dic:  dictionary

    Output:
    :returns: dictionary with missing values filled in.
    :rtype:   dictionary
    """
    RD = cp(dic)
    try:
        if isinstance(RD['dist'](),RD['dist']):
            RD['dist']=RD['dist']()
    except TypeError:
        pass
    except:
        raise AssertionError('Distribution %s does not seem to support a default distribution (called with no arguments) ' %(RD['dist']))
    if not RD['dist'].param.has_key('n'):
        n=1
    else:
        n=RD['dist']['n']
    if not 'nsamples' in dic.keys():
        RD['nsamples']=500000
    if not 'tolerance' in dic.keys():
        RD['tolerance']=1e-01
    if not 'support' in dic.keys():
        RD['support'] = (-np.inf,np.inf)
    if not 'proposal_low' in dic.keys():
        if RD['support'][1]-RD['support'][0]<np.inf:
            RD['proposal_low']=Uniform({'n':n,
                                    'low':RD['support'][0],
                                    'high':RD['support'][1]})
        elif RD['support'][0]==0 and RD['support'][1]==np.inf:
            RD['proposal_low']=Gamma({'u':1.,'s':2.})
        else:
            RD['proposal_low']=Gaussian({'n':n,'sigma':np.eye(n)*0.2})
    if not 'proposal_high'  in dic.keys():
        if RD['support'][1]-RD['support'][0]<np.inf:
            RD['proposal_high']=Uniform({'n':n,
                                    'low':RD['support'][0],
                                    'high':RD['support'][1]})
        elif RD['support'][0]==0 and RD['support'][1]==np.inf:
            RD['proposal_high']=Gamma({'u':3.,'s':2.})
        else:
            RD['proposal_high']= Gaussian({'n':n,'sigma':np.eye(n)*10})

    return RD
            
            
                                  
        

def check_loglik_importance(dic):
    """
    Checks a log-likelihood function via importance sampling. Nsamples
    are drawn from the proposal_high distribution (given in the
    dictionary). Therefore, it is assumed that the log-likelihood
    function of this distribution is correct. If also the likelihood
    function of the distribution under test (dic['dist']) is correct,
    the importance sampling estimate of the partition function, should
    yield a value close to 1. Here, an absolute error of
    dic['tolerance'] is accepted to pass the test. 

    Argument:
    :param dic: dictionary filled with (at least) proposal_high, nsamples, tolerance
    :type dic : dictionary

    """
    dic = fill_dict_with_defaults(dic)
    data = dic['proposal_high'].sample(dic['nsamples'])
    logQ = dic['proposal_high'].loglik(data)
    logP = dic['dist'].loglik(data)
    logZ = logsumexp(logP -logQ) - np.log(dic['nsamples'])
    diff = np.abs(np.exp(logZ)-1)
    assert np.abs(np.exp(logZ)-1)<dic['tolerance'], "Testing loglik failed for %s, difference is %g, which is bigger than %g, number of samples are: %d "%(dic['dist'].name,diff,dic['tolerance'],dic['nsamples'])

def check_sample(dic):
    """
    Checks a sample function via importance sampling. Nsamples are
    drawn from the distribution under test. Then the partition
    function of the proposal_low distribution is estimated using these
    samples and the log-likelihood function of the distribution under
    test. If sampling and both log-likelihood functions are correct,
    the importance sampling estimate should yield a partition function
    estimate of 1. Here, an absolute error of dic['tolerance'] is
    accepted to pass the test.

    Argument:
    :param dic: dictionary filled with (at least) proposal_low, nsamples, tolerance
    :type dic : dictionary

    """
    dic = fill_dict_with_defaults(dic)
    data = dic['dist'].sample(dic['nsamples'])
    logP = dic['proposal_low'].loglik(data)
    logQ = dic['dist'].loglik(data)
    logZ = logsumexp(logP -logQ) - np.log(dic['nsamples'])
    diff = np.abs(np.exp(logZ)-1)
    assert np.abs(np.exp(logZ)-1)<dic['tolerance'], "Testing sampling failed for %s, difference is %g, which is bigger than %g, number of smaples are: %d"%(dic['dist'].name,diff,dic['tolerance'],dic['nsamples'])


def check_array2primary(dic):
    """
    Checks, if the primary parameters of the distribution under test
    can be read out as an numpy array and read in again.

    Argument:
    :param dic: dictionary containing the distribution to test 
    :type dic : dictionary

    """

    dic = fill_dict_with_defaults(dic)
    d = dic['dist']
    params = d.primary
    OK = True
    arrorig = d.primary2array()
    if len(arrorig)>0:
        arr = arrorig+np.random.randn(*arrorig.shape)*0.001
        d.array2primary(arr)
        arrNew = d.primary2array()
        OK = OK and np.max(arrNew - arr)<1e-09
        for param in params:
            d.primary=[param]
            arrorig = d.primary2array()
            arr = arrorig+np.random.randn(*arrorig.shape)*0.001
            d.array2primary(arr)
            arrNew = d.primary2array()
            OK = OK and np.max(arrNew - arr)<1e-09
        assert OK
    else:
        assert True
    
def check_dldtheta(dic):
    """
    Checks the gradient wrt to the primary parmaters of the
    distribution under test, if such function is provided by that
    distribution. To this end a single data point is sampled and the
    gradient is compared with the finite difference approximation to
    the graident. An absolute error of the specified tolerance (within
    the dictionary) is allowed to pass the test.

    Argument:
    :param dic: dictionary containing the distribution to test, also tolerance has to be specified
    :type dic : dictionary

    """

    dic = fill_dict_with_defaults(dic)
    d = dic['dist']
    havedldtheta = True
    try:
        data = d.sample(1)
        theta0 = d.primary2array()
        d.dldtheta(data)
    except AbstractError:
        havedldtheta=False
    if havedldtheta:
        def f(X):
            d.array2primary(X)
            l = d.loglik(data)
            return np.sum(l)
        def df(X):
            d.array2primary(X)
            gv = d.dldtheta(data)
            return np.sum(gv,axis=1)
        theta0 = d.primary2array()
        err    = check_grad(f,df,theta0)
        assert err<dic['tolerance']
    else:
        assert True

def check_dldx(dic):
    """
    Checks the gradient wrt to the data of the
    distribution under test, if such function is provided by that
    distribution. To this end a single data point is sampled and the
    gradient is compared with the finite difference approximation to
    the graident. An absolute error of the specified tolerance (within
    the dictionary) is allowed to pass the test.

    Argument:
    :param dic: dictionary containing the distribution to test, also tolerance has to be specified
    :type dic : dictionary

    """

    dic = fill_dict_with_defaults(dic)
    d = dic['dist']
    havedldx=True
    try:
        data = d.sample(1)
        d.dldx(data)
    except AbstractError:
        havedldx=False
    if havedldx:
        data_copy = cp(data)
        def f(X):
            data_copy.X = X.reshape(data_copy.X.shape)
            return np.sum(d.loglik(data_copy))
        def df(X):
            data_copy.X = X.reshape(data_copy.X.shape)
            return d.dldx(data_copy)
        X0 = data.X.flatten()
        err = check_grad(f,df,X0)
        assert err<dic['tolerance']
    else:
        assert True

def check_default(dic):
    """
    Checks, if the specified distribution provides a proper
    distribution (as indicated by providing functions sample and
    loglik), if called with no arguments.

    Argument:
    :param dic: dictionary containing the distribution to test
    :type dic : dictionary
    
    """
    dic = fill_dict_with_defaults(dic)
    dist = dic['dist']
    try:
        if isinstance(dist,dist.__class__):
            dist = dist.__class__
    except AttributeError:
        pass
    dist =dist()
    have_sample =  hasattr(dist ,'sample')
    have_loglik =  hasattr(dist ,'loglik') 
    assert have_loglik and have_sample

def check_basic(dic):
    """
    Check if the distribution has basic functions such as sample and loglik
    
    Argument:
    :param dic: dictionary containing the distribution to test
    :type dic : dictionary

    """
    dic = fill_dict_with_defaults(dic)
    dist = dic['dist']
    have_sample =  hasattr(dist ,'sample')
    have_loglik =  hasattr(dist ,'loglik') 
    assert have_loglik and have_sample

    

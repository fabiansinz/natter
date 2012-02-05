from __future__ import division
from natter.Distributions import Gaussian
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

def test_loglik():
    for dic in distributions_to_test:
        yield check_loglik_importance,dic

def test_sample():
    for dic in distributions_to_test:
        yield check_sample, dic

def test_array2primary():
    for dic in distributions_to_test:
        yield check_array2primary, dic


def test_dldtheta():
    for dic in distributions_to_test:
        yield check_dldtheta, dic

def test_dldx():
    for dic in distributions_to_test:
        yield check_dldx,dic

def test_default():
    for dic in distributions_to_test:
        yield check_default,dic

def test_basic():
    for dic in distributions_to_test:
        yield check_basic,dic

        
def fill_dict_with_defaults(dic):
    RD = cp(dic)
    try:
        if isinstance(RD['dist'](),RD['dist']):
            RD['dist']=RD['dist']()
    except:
        pass
    if not 'nsamples' in dic.keys():
        RD['nsamples']=1000
    if not 'tolerance' in dic.keys():
        RD['tolerance']=1e-01
    if not 'support' in dic.keys():
        RD['support'] = (-np.inf,np.inf)
    if not 'proposal_low' in dic.keys():
        if RD['support'][1]-RD['support'][0]<np.inf:
            RD['proposal_low']=Uniform({'n':RD['dist']['n'],
                                    'low':RD['support'][0],
                                    'high':RD['support'][0]})
        elif RD['support'][0]==0 and RD['support'][0]==np.inf:
            RD['proposal_low']=Gamma({'u':1.,'s':2.})
        else:
            RD['proposal_low']= Gaussian({'n':RD['dist']['n'],'sigma':np.eye(RD['dist']['n'])*0.2})
    if not 'proposal_high'  in dic.keys():
        if RD['support'][1]-RD['support'][0]<np.inf:
            RD['proposal_high']=Uniform({'n':RD['dist']['n'],
                                    'low':RD['support'][0],
                                    'high':RD['support'][0]})
        elif RD['support'][0]==0 and RD['support'][0]==np.inf:
            RD['proposal_high']=Gamma({'u':3.,'s':2.})
        else:
            RD['proposal_high']= Gaussian({'n':RD['dist']['n'],'sigma':np.eye(RD['dist']['n'])*10})

    return RD
            
            
                                  
        

def check_loglik_importance(dic):
    dic = fill_dict_with_defaults(dic)
    data = dic['proposal_high'].sample(dic['nsamples'])
    logQ = dic['proposal_high'].loglik(data)
    logP = dic['dist'].loglik(data)
    logZ = logsumexp(logP -logQ) - np.log(dic['nsamples'])
    assert np.abs(np.exp(logZ)-1)<dic['tolerance']

def check_sample(dic):
    dic = fill_dict_with_defaults(dic)
    data = dic['dist'].sample(dic['nsamples'])
    logP = dic['proposal_low'].loglik(data)
    logQ = dic['dist'].loglik(data)
    logZ = logsumexp(logP -logQ) - np.log(dic['nsamples'])
    assert np.abs(np.exp(logZ)-1)<dic['tolerance']


def check_array2primary(dic):
    dic = fill_dict_with_defaults(dic)
    d = dic['dist']
    params = d.primary
    OK = True
    arrorig = d.primary2array()
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

    
def check_dldtheta(dic):
    dic = fill_dict_with_defaults(dic)
    d = dic['dist']
    data = d.sample(1)
    theta0 = d.primary2array()
    havedldtheta = True
    try:
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
    dic = fill_dict_with_defaults(dic)
    d = dic['dist']
    data = d.sample(1)
    havedldx=True
    try:
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
    dic = fill_dict_with_defaults(dic)
    if 'basedist' in dic.keys():
        dist = dic['basedist']
        dist =dist()
        have_sample =  hasattr(dist ,'sample')
        have_loglik =  hasattr(dist ,'loglik') 
        assert have_loglik and have_sample
    else:
        assert True

def check_basic(dic):
    """
    Check if the distribution has basic functions such as sample and loglik
    """
    dic = fill_dict_with_defaults(dic)
    dist = dic['dist']
    have_sample =  hasattr(dist ,'sample')
    have_loglik =  hasattr(dist ,'loglik') 
    assert have_loglik and have_sample

    

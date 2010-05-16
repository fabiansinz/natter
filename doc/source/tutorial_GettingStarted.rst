Getting started
================

The basic structure of the *natter* is simple. It contains a module
for data, one for Transformations on the data, one module for
distributions and, finally, a module that keeps all the auxiliary
stuff which is used in the other modules. 

The first and easiest thing we can do is to create a data object and
store some data in it. 

>>> from numpy.random import randn
>>> from natter import DataModule
>>> X = randn(3,10)
>>> dat = DataModule.Data(X,'My Data')
>>> print dat
------------------------------
Data object: My Data
        10  Examples
        3  Dimensions
------------------------------

A faster way would have been to use the built-in function gauss in the
DataModule.


>>> dat = DataModule.DataSampler.gauss(3,10000)
>>> dat
------------------------------
Data object: Multivariate Gaussian data.
        10000  Examples
        3  Dimensions
------------------------------

In order to make it more interesting, we will give the data some
covariance. We can look at it by using the *cov* function of the Data
object.

>>> from numpy import dot
>>> S = randn(3,3)
>>> S = dot(S,S.T)
>>> dat = DataModule.DataSampler.gauss(3,10000,sigma=S)
>>> dat.cov()

Now it is time to create a Distribution object and fit it to the
data. Since the data is Gaussian, we will create a Gaussian
distribution. 

>>> from natter import Distributions
>>> p = Distributions.Gaussian({'n':3})
>>> p
------------------------------
Gaussian Distribution
        n: 3
        mu:
        [ 0.  0.  0.]
        sigma:
        [[ 1.  0.  0.]
         [ 0.  1.  0.]
         [ 0.  0.  1.]]
------------------------------

This demonstrates the generic way in which every distribution is
created. You either call it with no arguments or with a dictionary
that contains the intial parameters for the distribution. Check out
the documentation of each single distribution find out the parameters
of each particular distribution. In case of the Gaussian, the
parametes are the dimensionality *n*, the covariance matrix *sigma*
and the mean *mu*. You do not need to specify all of them when
creating a distribution. Just specify the ones that you want to set on
your own. It is important that the initial parameters are always
passed as a dictionary with the keys set to the parameter names of
that distribution. Internally, the parameters are stored in the member
called *param*


>>> p.param
{'mu': array([ 0.,  0.,  0.]), 'sigma': array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]]), 'n': 3}


Let's fit the Gaussian to our sampled data. Each distribution has a
method called *estimate*. When implemented, it estimates the
parameters of that distribution on the data you give it.

>>> p.estimate(dat)


Of course, we can also sample from our distribution.

>>> dat2 = p.sample(50000)





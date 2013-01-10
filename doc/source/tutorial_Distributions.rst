Tutorial on the Distributions module
====================================

Creating distribution objects
-----------------------------

Let's start with how to pass parameters to distributions at
construction time. Take a simple univariate distribution

>>> from natter import Distributions
>>> p = Distributions.Gamma()
------------------------------
Gamma Distribution
	s: 1.0
	u: 1.0
	Primary Parameters:[u, s]
------------------------------

Almost all distributions have a default constructor. This means you
can simply generate a distribution with no parameters and it will give
you a fully functional object. Only in some cases, in which an
informed choice of default parameters is not possible, creating a
distribution with an empty parameter list will fail. 

The distribution object will display the parameters that determine its
current state in the prompt. If you want to pass different parameter
when constructing the distribution you can do that in two different
ways. The easiest is with name parameter, just like in python

>>> p = Distributions.Gamma(u=1.0,s=3.0)
------------------------------
Gamma Distribution
	s: 3.0
	u: 1.0
	Primary Parameters:[u, s]
------------------------------

Even though we do some parameter checking when the distribution is
created it is not very extensive. Therefore, it is recommendable that
you make sure your parameters are correct and have the correct type. 

For example, you could create a Gamma distribution with negative shape
parameters. Although the constructor does not complain, this does not
make any sense mathematically. So make sure that your parameters are
chosen correctly.

The other way to create a distribution with specified parameters is
with a dictionary that contains their names and values as key-value
pairs

>>> p = Distributions.Gamma({'u':1.2, 's':2.})
------------------------------
Gamma Distribution
	s: 2.0
	u: 1.2
	Primary Parameters:[u, s]
------------------------------

Since the important parameters are stored in a field called 'param' in
the distribution

>>> p.param
{'s': 2.0, 'u': 1.2}

this gives you an easy way to transfer parameters from one
distribution to the next and store the state of a particular
distribution efficiently.

>>> pa = p.param
>>> q = Distributions.Gamma(pa)
------------------------------
Gamma Distribution
	s: 2.0
	u: 1.2
	Primary Parameters:[u, s]
------------------------------

Of course, this would have worked as well

>>> q = Distributions.Gamma(**pa)

Changing parameters
-------------------

Of course, parameters can be accessed via the 'param' dictionary, but
there are simpler ways to do so: simply treat the distribution as a
dictionary for the parameters.

>>> q['s']
2.0
>>> q['s'] = 3.0
------------------------------
Gamma Distribution
	s: 3.0
	u: 1.2
	Primary Parameters:[u, s]
------------------------------

Sampling data and (log)-likelihood
----------------------------------

Sampling data from a distribution is easy: Simply call the sample
method with the number of data points you want.

>>> p = Distributions.Gamma()
>>> dat = p.sample(10)
------------------------------
Data object: 10 samples from Gamma Distribution
	10  Examples
	1  Dimension(s)
------------------------------

The probability assigned to each data point by that distribution can
be assessed via the pdf method

>>> p.pdf(dat)
array([ 0.82412216,  0.08277057,  0.9910718 ,  0.61119109,  0.26334754,
        0.39543065,  0.51351136,  0.47248581,  0.04547772,  0.3233271
	])

The log-likelihood is obtained via loglik

>>> p.loglik(dat)
p.loglik(dat)
array([-0.1934365 , -2.49168276, -0.0089683 , -0.49234561, -1.33428066,
       -0.92777986, -0.66648312, -0.74974757, -3.09053284, -1.12909079])


Because this occurs so often, there is also a shortcut for that

>>> p(dat)
array([-0.1934365 , -2.49168276, -0.0089683 , -0.49234561, -1.33428066,
       -0.92777986, -0.66648312, -0.74974757, -3.09053284, -1.12909079])



Primary parameters
------------------

You will have noticed that the prompt representation of the
distribution also contains a list of parameter names called primary
parameters. These parameter are the ones that can be estimated from
data. 

For illustration, let's take a look at a more complex distribution

>>> p = Distributions.LpSphericallySymmetric(n=2)
------------------------------
Lp-Spherically Symmetric Distribution
	p: 2.0
	n: 2
	rp: 
	------------------------------
	Gamma Distribution
		s: 1.0
		u: 1.0
		Primary Parameters:[u, s]
	------------------------------
	Primary Parameters:[rp, p]
------------------------------

You can see that this distribution has three parameters: 'p', 'n', and
'rp'. 'p' is a regular parameter, 'n' is the dimensionality of the
data the distribution is defined on and 'rp' is another distribution
object (the radial distribution of the Lp-spherically symmetric
distribution).

Now obviously it does not make sense to fit 'n' to data. Therefore,
'n' is not included in the list of primary parameters. The other two,
however, can be estimated from data and that's the reason why their
names are listed under the primary parameters. 

Within the distribution object, the primary parameters are simply a
list of keys into the param dictionary.

>>> p.primary
['rp', 'p']

You can delete elements from that list. In this case, these parameters
should not be estimated when calling the 'estimate' in the
object. Let's take a look at that in more detail. 

First we sample some data from the current distribution

>>> dat = p.sample(5000)
------------------------------
Data object: Samples from Lp-Spherically Symmetric Distribution
	5000  Examples
	2  Dimension(s)
History:
    |-sampled 5000 examples from Lp-generalized Normal
    |-normalized Data with p=2.0
    |-Scaled  with 5000 samples from Gamma Distribution
------------------------------

Now let's change the value of some of the parameters

>>> p['p'] = 1.0
>>> p['rp'] = Distributions.Gamma(s=4.0)
------------------------------
Lp-Spherically Symmetric Distribution
	p: 1.0
	n: 2
	rp: 
	------------------------------
	Gamma Distribution
		s: 4.0
		u: 1.0
		Primary Parameters:[u, s]
	------------------------------
	Primary Parameters:[rp, p]
------------------------------

.. note:: It is not possible to set the parameters of the radial
   distribution via 

   >>> p['rp']['s'] = 4.0 

   due to the way the __getitem__ methods are implemented.

Now, let's fit this distribution to the data we sample before.

>>> p.estimate(dat)
------------------------------
Lp-Spherically Symmetric Distribution
	p: 1.79674102795
	n: 2
	rp: 
	------------------------------
	Gamma Distribution
		s: 0.997205657091
		u: 0.998633335022
		Primary Parameters:[u, s]
	------------------------------
	Primary Parameters:[rp, p]
------------------------------

You can see that the parameters have been changed back to almost their
original values (the ones that we sampled the data with). Only the 'p'
is not quite 2 anymore. This is because the log-likelihood can be
quite flat for a large range of values for p and, therefore, also a p
around 1.8 will do. 

But imagine now that we know that p=2. In that case we don't want the
estimation procedure to fiddle around with it. This is the situation
where primary parameters come in handy. 

First, we set p to the value we want it to have.

>>> p['p'] = 2.0

For demonstration purposes, we also change the radial distribution
again

>>> p['rp'] = Distributions.Gamma(s=4.0)
------------------------------
Lp-Spherically Symmetric Distribution
	p: 2.0
	n: 2
	rp: 
	------------------------------
	Gamma Distribution
		s: 4.0
		u: 2.0
		Primary Parameters:[u, s]
	------------------------------
	Primary Parameters:[rp, p]
------------------------------

Now, before we fit it to data, we remove 'p' from the list of primary
parameters

>>> p.primary.remove('p')
------------------------------
Lp-Spherically Symmetric Distribution
	p: 2.0
	n: 2
	rp: 
	------------------------------
	Gamma Distribution
		s: 4.0
		u: 2.0
		Primary Parameters:[u, s]
	------------------------------
	Primary Parameters:[rp]
------------------------------

Afterwards, we fit the distribution to data again

>>> p.estimate(dat)
------------------------------
Lp-Spherically Symmetric Distribution
	p: 2.0
	n: 2
	rp: 
	------------------------------
	Gamma Distribution
		s: 0.975559481182
		u: 0.998459686353
		Primary Parameters:[u, s]
	------------------------------
	Primary Parameters:[rp]
------------------------------

You can see that 'p' stays unchanged because it was not in the primary
parameters list anymore. 

.. note:: Although it is possible to add primary parameters to the
   list, the distribution object won't know what to do with it. We
   don't check whether the primary parameters are correct. This is
   your responsibility.

Gradients
---------

Primary parameters make it easy to create fitting routines for them
via gradient ascent. As we explain in the accompanying paper, we
intend the natter to be flexible and easily extendible. In many cases,
the parameters of a distribution are simply estimated via gradient
ascent on the log-likelihood. It would be a waste of coding time to
implement the same routine over and over again. If you think about it,
all that this routine needs is the gradient of the log-likelihood
w.r.t. the primary parameters on the data. In order to work with it,
the gradient ascent methods also needs the parameters in array
form. All this is accomplished by the dldtheta, array2primary, and
primary2array methods that we will discuss now. 

Let's take a look at an easier distribution again

>>> p = Distributions.Gamma()
------------------------------
Gamma Distribution
	s: 1.0
	u: 1.0
	Primary Parameters:[u, s]
------------------------------

Let's assume you just implemented this distribution object and don't
have any estimation routine so far. However, you want to do it via
gradient ascent. The first thing to do is to implement the
'primary2array' method that forms an array containing the primary
parameters

>>> a = p.primary2array()
>>> array([ 1.,  1.])

Notice that it yields the primary parameters exactly in the order in
which they are specified in the 'primary' list and does not return
them if they are not part of the 'primary' list.

>>> p['u'] = 2.
>>> p.primary2array()
array([ 2.,  1.])
>>> p.primary = ['s','u']
>>> p.primary2array()
p.primary = ['s','u']
>>> p.primary = ['s']
>>> p.primary2array()
p.primary = ['s']

Of course, the gradient method will also need a way to reverse this
operation. To that end, you will have to implement the 'array2primary'
method that does exactly that. Notice that also this method respects
the order and the elements in the primary array.

>>> from numpy import *
>>> p = Distributions.Gamma()
>>> a = array([2.3, 1.3])
>>> p.array2primary(a)
------------------------------
Gamma Distribution
	s: 
	[ 1.3]
	u: 
	[ 2.3]
	Primary Parameters:[u, s]
------------------------------

Now, the (almost) final method that natter needs for parameter
optimization is the method that returns the gradients on data. This is
the dldtheta method.

>>> p = Distributions.Gamma()
>>> dat = p.sample(10)
>>> gr = p.dldtheta(dat)
array([[-0.45949552, -0.10639768, -0.3317563 , -2.90575816,  0.81977497,
        -0.89446596, -0.38410357, -1.54950748, -0.28445376,  1.14863712],
       [-0.64538096, -0.49521028, -0.59706176, -0.96928407,  0.27450683,
        -0.77046084, -0.61761191, -0.88077266, -0.57754376,  0.77078235]])

It returns an array with the derivative of the log-likelihood
w.r.t. to the primary parameter on every data point. 

Note that this method also needs to respect the order and the elements
of the primary list.

This is almost all we need. The default method for estimate (inherited
from Distributions.Distribution) get the parameters via primary2array,
runs a gradient descent on the log-likelihood via the gradients from
dldtheta (you also need to implement loglik or pdf for that) and sets
them afterwards via array2primary.

There is potentially one thing missing. If the parameters are only
defined on a certain range, the optimizer should know about that. You
can tell it that via the primaryBounds methods that returns a list of
tuples with bounds for each primary parameter. If the parameter is not
bounded or only half-bounded, simply return None for that value.

>>> p = Distributions.Gamma()
>>> p.primaryBounds()
>>> [(1e-06, None), (1e-06, None)]

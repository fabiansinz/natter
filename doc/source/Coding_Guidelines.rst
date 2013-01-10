Mandatory Coding Guidelines (aka the Gerwinn-Sinz Coding Agreement)
===================================================================

1. **Parameters of Distributions 1** Parameters of distributions are
set via the __setitem__ and __getitem__ methods. By default,
__getitem__ uses the *parameters* method of the distribution object to
return the respective parameter. If the key is not a parameter of the
distribution, it raises a KeyError. The method __setitem__ calls the
__setitem__ method of a field param, which is a dictionary containing
all the parameters of the distribution (like in the intialization). If
not all parameters of the distribution are in the *param* dictionary,
the __setitem__ method has to be overwritten.

2. **Parameters of Distributions 2** Every distribution object must
implement a *parameters* function that returns (when called with no
parameters) a dictionary that has the names of the parameters of that
distribution as keys and *deepcopies* of the parameters as
values. When passed to a contructor of the same type of distribution,
this dictionary must be enough to create a mathematically equivalent
distribution object.

3. **Parameters of Distribution 3** Distribution constructors must be
compatible to be called in the following ways

>>> d = mydistribution(n=3,p=3)

>>> d = mydistribution({'n':3,'p':3})

>>> d = mydistribution(n=3,param={'p':3})

>>> d = mydistribution(param={'p':3})

(see also natter.Utils.parseParameters). This basically means that a
constructor may at least have one single element in args, which is the
dictionary param. Other key value pairs of param may be passed via the
"n=3" syntax.

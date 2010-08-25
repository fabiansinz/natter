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

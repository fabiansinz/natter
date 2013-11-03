PCauchy distribution
====================


The p-Cauchy distribution in n dimensions is the ratio distribution of
a Lp-spherically symmetric distribution in n+1 dimensions (see [Szablowski1998]_).

.. math::

   p(\boldsymbol{y}) = \frac{p^{n}\Gamma\left(\frac{n+1}{p}\right)}{2^{n}\Gamma^{n+1}\left(\frac{1}{p}\right)\left(1+\|\boldsymbol{y}\|_{p}^{p}\right)^{\frac{n+1}{p}}}

:math:`p >0` refers to the shape parameter.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.PCauchy
   :members:

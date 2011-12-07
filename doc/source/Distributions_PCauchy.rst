PCauchy distribution
====================


The p-Cauchy distribution in n dimensions is the ratio distribution of
a Lp-spherically symmetric distribution in n+1 dimensions.  

.. math::

   p(\boldsymbol{y}) = \frac{p^{n}\Gamma\left(\frac{n+1}{p}\right)}{2^{n}\Gamma^{n+1}\left(\frac{1}{p}\right)\left(1+\|\boldsymbol{y}\|_{p}^{p}\right)^{\frac{n+1}{p}}}



.. autoclass:: natter.Distributions.PCauchy
   :members: loglik, pdf, estimate, all, copy, parameters, dldtheta 
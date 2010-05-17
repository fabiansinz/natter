GammaP distribution
===================

Univariate distribution for which :math: x^p is Gamma distributed.

.. math::

   p(x) =  \frac{p\cdot x^{u+p-2}}{\Gamma(u)\,s^u} \exp{\left(-\frac{x^p}{s}\right)}



.. autoclass:: natter.Distributions.GammaP
   :members: __init__,sample, loglik, pdf, cdf, ppf, dldx, estimate, all
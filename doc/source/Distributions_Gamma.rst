Gamma distribution
===================

The gamma distribution is a univariate density with the p.d.f.

.. math::

   p(x) =  \frac{x^{u-1}}{\Gamma(u)\,s^u} \exp{\left(-\frac{x}{s}\right)}



.. autoclass:: natter.Distributions.Gamma
   :members: __init__,sample, loglik, pdf, cdf, ppf, dldx, estimate, all

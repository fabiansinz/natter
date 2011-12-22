GammaP distribution
===================

Univariate distribution for which :math: x^p is Gamma distributed.

.. math::

   p(x) =  \frac{p\cdot x^{up-1}}{\Gamma(u)\,s^u} \exp{\left(-\frac{x^p}{s}\right)}



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.GammaP
   :members: sample, loglik, pdf, cdf, ppf, dldx, estimate, all, copy, parameters

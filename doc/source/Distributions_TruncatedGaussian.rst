TruncatedGaussian distribution
==============================

The TruncatedGaussian distribution is a univariate density on the interval
[a,b] with the p.d.f.

.. math::

   p(x) = \frac{\frac{1}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right)}{\Phi\left(\frac{b-\mu}{\sigma}\right)- \Phi\left(\frac{a-\mu}{\sigma}\right)}



.. autoclass:: natter.Distributions.TruncatedGaussian
   :members: sample, loglik, pdf, cdf, ppf, dldtheta, estimate, all, copy, parameters

Multivariate Gaussian Distribution
==================================

The multivariate Gaussian distribution is given by

.. math::

   p(\boldsymbol{x}) = \frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}}\, e^{ -\frac{1}{2}(\boldsymbol{x}-\mu)'\Sigma^{-1}(\boldsymbol{x}-\mu) }



.. moduleauthor:: Sebastian Gerwinn <sebastian@bethgelab.org>
.. autoclass:: natter.Distributions.Gaussian
   :members: sample, loglik, pdf, cdf, ppf, dldx, estimate, all, copy, parameters

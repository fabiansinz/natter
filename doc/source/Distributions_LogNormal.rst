Log-Normal distribution
=======================

The distribution object implements the log-normal distribution with
the parametrization 

.. math::

   p(x)=\frac{1}{x\sqrt{2\pi\sigma^{2}}}\exp\left(-\frac{(\log x-\mu)^{2}}{2\sigma^{2}}\right)

:math:`\mu` and :math:`\sigma` are the mean and the standard deviation of the underlying Gaussian.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.LogNormal
   :members: sample, loglik, pdf, cdf, ppf, dldx, estimate, all, copy, parameters

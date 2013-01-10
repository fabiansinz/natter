LogLogistic distribution
========================

The LogLogistic distribution is a univariate density on with the pdf

.. math::

   p(x|\alpha,\beta)=\frac{(\beta/\alpha)(x/\alpha)^{\beta-1}}{\left(1+(x/\alpha)^{\beta}\right)^{2}}.


where :math:`\alpha` and :math:`\beta` the scale and shape parameters.



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.LogLogistic
   :members: sample, loglik, pdf, cdf, estimate

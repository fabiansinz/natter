LogLogistic distribution
========================

The LogLogistic distribution is a univariate density on with the pdf

.. math::

   p(x|\alpha,\beta)=\frac{(\beta/\alpha)(x/\alpha)^{\beta-1}}{\left(1+(x/\alpha)^{\beta}\right)^{2}}.


where :math:`\alpha > 0` and :math:`\beta > 0` the scale and shape parameters.



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.LogLogistic
   :members:

Transformed Distribution
========================

The Transformed distribution is a univariate density with the pdf

.. math::

   p(y) = q(f^{-1}(y))\left| \frac{df^{-1}}{dy} \right|


.. autoclass:: natter.Distributions.Transformed
   :members: sample, loglik, pdf, cdf, ppf,  all, copy, parameters

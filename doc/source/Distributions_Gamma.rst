Gamma distribution
===================

The gamma distribution is a univariate density with the p.d.f.

.. math::

   p(x) =  \frac{x^{u-1}}{\Gamma(u)\,s^u} \exp{\left(-\frac{x}{s}\right)}

where  :math:`u > 0` refers to the shape parameter and :math:`s>0` to the scale parameter.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.Gamma
   :members: primaryBounds, sample, loglik, pdf, cdf, ppf, dldtheta, dldx, estimate

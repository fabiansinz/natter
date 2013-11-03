GammaP distribution
===================

Univariate generalized gamma distribution for which :math: x^p is Gamma distributed.

.. math::

   p(x) =  \frac{p\cdot x^{up-1}}{\Gamma(u)\,s^u} \exp{\left(-\frac{x^p}{s}\right)}

where :math:`u,p > 0` refer to the shape parameters, and :math:`s>0` is the scale parameter.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.GammaP
   :members: parameters, loglik, dldx, dldtheta, d2ldtheta2, pdf, cdf, ppf, sample, estimate, primaryBounds

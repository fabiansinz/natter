NakaRushton distribution
========================

The NakaRushton distribution is a univariate density on the interval
[0,B] with the p.d.f.

.. math::

   p(r) =  \frac{p\kappa^{n}\sigma^{2}r^{n-1}}{\mathfrak{G}\left(\frac{n}{p},\frac{\kappa^{p}}{2s}\right)\Gamma\left(\frac{n}{p}\right)\left(2s\right)^{\frac{n}{p}}\left(\sigma^2+r^2\right)^{\frac{n+2}{2}}}\exp\left(-\frac{\kappa^{p}r^{p}}{2s\left(\sigma^2+r^2\right)^{\frac{p}{2}}}\right)

where :math:`\mathfrak{G}` is the regularized incomplete gamma function.

.. autoclass:: natter.Distributions.NakaRushton
   :members: sample, loglik, pdf, cdf, dldtheta, estimate, all, copy, parameters

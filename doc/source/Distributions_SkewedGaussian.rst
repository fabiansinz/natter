SkewedGaussian distribution
===========================

The SkewedGaussian distribution is a univariate density on with the pdf

.. math::

   p(x) =
   \frac{2}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right)\Phi\left(\alpha\frac{x-\mu}{\sigma}\right)

where :math:`\phi` and :math:`\Phi` are the p.d.f. and the c.d.f. of
the standard normal distribution. 



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.SkewedGaussian
   :members:

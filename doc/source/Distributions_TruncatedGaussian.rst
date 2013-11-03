TruncatedGaussian distribution
==============================

The TruncatedGaussian distribution is a univariate density on the interval
[a,b] with the p.d.f.

.. math::

   p(x) = \frac{\frac{1}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right)}{\Phi\left(\frac{b-\mu}{\sigma}\right)- \Phi\left(\frac{a-\mu}{\sigma}\right)}



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.TruncatedGaussian
   :members:

TruncatedExponentialPower distribution
======================================

The TruncatedExponentialPower distribution is a univariate density on the interval
[a,b] with the p.d.f.

.. math::

   p(x) = \frac{p\exp\left(-\frac{\vert x\vert^{p}}{s}\right)}{2s^{\frac{1}{p}}\Gamma\left(\frac{1}{p}\right)\left(\Xi(b)-\Xi(a)\right)}

where :math:`\Xi` is the c.d.f. of the ExponentialPower distribution. 

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.TruncatedExponentialPower
   :members:

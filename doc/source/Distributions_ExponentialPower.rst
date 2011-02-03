Exponential Power Distribution
==============================

This distribution class implements the exponential power distribution 

.. math::

   p(x) = \frac{p}{2s^\frac{1}{p} \Gamma\left[\frac{1}{p}\right]} \exp\left(-\frac{|x|^p}{s}\right)

.. autoclass:: natter.Distributions.ExponentialPower
   :members: sample, loglik, pdf, dldx, estimate, all, copy, parameters

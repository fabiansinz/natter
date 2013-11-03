Exponential Power Distribution
==============================

This distribution class implements the exponential power distribution 

.. math::

   p(x) = \frac{p}{2s^\frac{1}{p} \Gamma\left[\frac{1}{p}\right]} \exp\left(-\frac{|x|^p}{s}\right)

where  :math:`p > 0` refers to the shape parameter and :math:`s>0` to the scale parameter.


.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.ExponentialPower
   :members:

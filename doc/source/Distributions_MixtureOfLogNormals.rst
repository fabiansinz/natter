Univariate Mixture of Log-Normal Distributions
==============================================

The mixture of log-normal distributions is given by

.. math::

   p(x) = \sum_{k=1}^K \pi_k \frac{1}{\sqrt{(2\pi\sigma^2)} \cdot x}\, \exp\left( -\frac{1}{2\sigma^2}(\log x-\mu)^2 \right)

The mixture weights need to sum to one, i.e.
:math:`\sum_{i=1}^K \pi_i=1` and :math:`\pi_i \ge 0`.


.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.MixtureOfLogNormals
   :members:

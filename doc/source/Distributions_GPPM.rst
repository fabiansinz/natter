Gaussian Product Process Model distribution
===========================================

The distribution object implements the distribution corresponding to a
Gaussian Process Product model with  the parametrization

.. math::
   `y(x) = f(x) g(x) \exp(h(x)) + \epsilon`
   
    where f,g,h are Gaussian processes over the possibly two-dimensional pixel space.
    However, it is assumed, that the xs are the same for all data points.


.. autoclass:: natter.Distributions.GPPM
   :members: sample, loglik, pdf, estimate

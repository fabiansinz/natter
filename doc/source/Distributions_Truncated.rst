Truncated Distribution
======================

The Truncated distribution is a univariate density on the interval
[a,b] with the p.d.f.

.. math::

   p(x) = \frac{q(x)}{F_q(b)-F_q(a)}

where :math:`F_q` is the c.d.f. of the base distribution q (the one
that is truncated).

.. autoclass:: natter.Distributions.Truncated
   :members: sample, loglik, pdf, cdf, ppf,  all, copy, parameters

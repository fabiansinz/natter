Truncated Distribution
======================

The Truncated distribution is a univariate density on the interval
[a,b] with the p.d.f.

.. math::

   p(x) = \frac{q(x)}{F_q(b)-F_q(a)}

where F denotes the c.d.f. of the base distribution q (the one
that is truncated).

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.Truncated
   :members:

Kumaraswamy distribution
========================

The Kumaraswamy distribution is a univariate density on the interval
[0,B] with the p.d.f.

.. math::

   p(x) =  \frac{ab}{B^{ab}} x^{a-1} (B^a-x^a)^{b-1}



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.Kumaraswamy
   :members: sample, loglik, pdf, cdf, ppf, dldtheta, estimate, all, copy, parameters

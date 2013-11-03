Finite Mixture of arbitrary distributions
=========================================

Base class for a finite mixture of base distributions.

.. math::

   p(x|\theta) = \sum_{k=1}^{K} \alpha_k p(x|\theta_k)

s.t. :math:`\sum_{i=1}^K \alpha_i=1` and :math:`\alpha_i \ge 0`.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. moduleauthor:: Sebastian Gerwinn <sebastian@bethgelab.org>
.. autoclass:: natter.Distributions.FiniteMixtureDistribution
   :members: estimate, parameters, sample, loglik, pdf, ppf, primary2array, primaryBounds, array2primary, dldtheta, cdf, mixturePosterior,

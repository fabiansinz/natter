Finite Mixture of arbitrary distributions
=========================================

Base class for a finite mixture of base distributions.

.. math::

   p(x|\theta) = \sum_{k=1}^{K} \alpha_k p(x|\theta_k)



.. autoclass:: natter.Distributions.FiniteMixtureDistribution
   :members: estimate, parameters

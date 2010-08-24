Mixture of Dirichlet Distributions
==================================

The mixture of Dirichlet distributions has the form

.. math::

   p(\boldsymbol{x}) = \sum_{k=1}^K \pi_k \frac{\Gamma\left[\sum_{i=1}^n
   \alpha_{ik}\right]}{\prod_{i=1}^n\Gamma\left[\alpha_{ik}\right]}
   \prod_{i=1}^n x_i^{\alpha_{ik}-1}

s.t. :math:`\sum_{i=1}^n x_i=1`, :math:`x_i > 0` and :math:`\alpha_i
\ge 0`.


.. autoclass:: natter.Distributions.MixtureOfDirichlet
   :members: sample, loglik, pdf, estimate, all, copy, getPosteriorWeights, parameters

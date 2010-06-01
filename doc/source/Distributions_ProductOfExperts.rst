Product Of Experts
==================

The product of experts distribution has the form 

.. math::

   p(\boldsymbol{x}) = \frac{1}{Z} \prod_{k=1}^m \psi_k(\boldsymbol{w}^\top\boldsymbol{x})

where :math:`\psi_k` are univariate experts, i.e. factors that shape the distribution in direction :math:`\boldsymbol{w}`.


.. autoclass:: natter.Distributions.ProductOfExperts
   :members: sample, loglik, pdf, estimate, all, copy, dldx

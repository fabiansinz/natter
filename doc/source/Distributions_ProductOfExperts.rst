Product Of Experts
==================

The product of experts distribution has the form 

.. math::

   p(\boldsymbol{x}) = \frac{1}{Z} \prod_{k=1}^m \psi_k(\boldsymbol{w}_{k}^\top\boldsymbol{x})^{\alpha_k}

where :math:`\psi_k` are univariate experts, i.e. factors that shape the distribution in direction :math:`\boldsymbol{w}`.


.. autoclass:: natter.Distributions.ProductOfExperts
   :members: estimate, copy, parameters

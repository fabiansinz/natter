The complete linear model
=========================

The density of the complete linear model has the form 

.. math::

   p(\boldsymbol{x}) = q(\boldsymbol{W} \boldsymbol{x})

where :math:`\boldsymbol{W}` is a orthogonal matrix,
i.e. :math:`\boldsymbol{W}\in SO(n)`.

.. autoclass:: natter.Distributions.CompleteLinearModel
   :members: __init__,sample, loglik, pdf, dldx, estimate, all, copy

Complete linear model
=====================

The density of the complete linear model has the form

.. math::

   p(\boldsymbol{x}) = q(\boldsymbol{W} \boldsymbol{x})

where :math:`\boldsymbol{W}\in SO(n)` is a orthogonal matrix, and :math:`q` is a base distribution represented by
another Distribution object.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.CompleteLinearModel
   :members: parameters, loglik, estimate, sample, objective

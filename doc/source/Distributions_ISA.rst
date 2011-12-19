Independent Subspace Analysis (ISA)
===================================

This distribution class implements the independent subspace
model with the joint distribution

.. math::

   p(\boldsymbol{x}) = \prod_{k=1}^K p_k(\boldsymbol x_{I_k})
   
where :math:`I_k` are index lists into disjoint subspaces,
i.e. :math:`I_k\cap I_j=\emptyset` for :math:`k\not= j`

The ISA class does not implement filters. If you wish to do that
create a CompleteLinearModel with ISA as a base class.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.ISA
   :members: sample, loglik, pdf, dldx, estimate, all, copy, parameters

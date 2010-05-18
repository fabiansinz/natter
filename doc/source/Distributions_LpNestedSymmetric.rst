:math:`L_p`-nested symmetric distribution
=========================================

The distribution object implements the general class of
:math:`L_p`-nested symmetric distribution which has the general
form (see [SinzBethge2010]_)

.. math::
   p(\boldsymbol{x}) =
   \frac{\varrho(f(\boldsymbol{x}))}{f(\boldsymbol{x})^{n-1}\mathcal
   S_f(1)}=
   \frac{\varrho(f(\boldsymbol{x}))}{2^n f(\boldsymbol{x})^{n-1}}\prod_{I\in \mathcal   I}p_I^{\ell_I-1}\prod_{k=1}^{\ell_I-1} B\left[\frac{\sum_{i=1}^{k} n_{I,k}}{p_I},\frac{n_{I,k+1}}{p_I}\right]^{-1}

:math:`\varrho` is any radial density on :math:`\mathbb R_+` and
:math:`f` is an :math:`L_p`-nested function (see
:doc:`LpNestedFunction <Auxiliary_LpNestedFunction>`). In the *natter*
the radial density is specified via the parameter 'rp' which is
another Distribution object.

.. autoclass:: natter.Distributions.LpNestedSymmetric
   :members: sample, loglik, pdf, dldx, estimate, all, copy

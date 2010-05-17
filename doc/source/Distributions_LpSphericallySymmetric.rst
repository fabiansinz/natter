:math:`L_p`-spherically symmetric distribution
==============================================

The distribution object implements the general class of
:math:`L_p`-spherically symmetric distribution which has the general
form (see [GuptaSong1997]_)

.. math::

   p(\boldsymbol{x})=\frac{p^{n-1}\Gamma\left(\frac{n}{p}\right)}{\|\boldsymbol{x}\|_{p}^{n-1}2^{n}\Gamma^{n}\left(\frac{1}{p}\right)}\varrho\left(\|\boldsymbol{x}\|_{p}\right)


:math:`\varrho` is any radial density on :math:`\mathbb R_+`. In the
*natter* it is specified via the parameter 'rp' which is another
Distribution object.

.. autoclass:: natter.Distributions.LpSphericallySymmetric
   :members: __init__,sample, loglik, pdf, dldx, estimate, all, logSurfacePSphere, copy

Dirichlet Distribution
======================

The density of the Dirichlet distribution has the form

.. math::

   p(\boldsymbol{x}) = \frac{\Gamma\left[\sum_{i=1}^n
   \alpha_i\right]}{\prod_{i=1}^n\Gamma\left[\alpha_i\right]}
   \prod_{i=1}^n x_i^{\alpha_i-1}

s.t. :math:`\sum_{i=1}^n x_i=1`, :math:`x_i > 0` and :math:`\alpha_i
\ge 0`.


.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.Dirichlet
   :members: sample, loglik, pdf, dldx, estimate, all, copy, parameters

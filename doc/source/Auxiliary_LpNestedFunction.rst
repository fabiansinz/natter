The LpNestedFunction class
==========================

The LpNestedFunction submodule represents functions like 

.. math::
   f(\boldsymbol{x}) = \left(\left(\sum_{i=1}^{n_1}|x_i|^{p_1}\right)^\frac{p_\emptyset}{p_1} + ... +\left(\sum_{i=n_1+...+n_{\ell-1}+1}^{n}|x_i|^{p_\ell}\right)^\frac{p_\emptyset}{p_\ell}\right)^\frac{1}{p_\emptyset}

Their general form can be visualized like a tree. Every node computes
the :math:`L_p`-norm of its children. The values of the children nodes
can either be coefficients of :math:`\boldsymbol{x}` or outcomes of
:math:`L_p`-functions themselves (see [SinzEtAl2009]_).

   


.. autoclass:: natter.Auxiliary.LpNestedFunction
   :members: __init__,f, dfdx,i, plotGraph, copy




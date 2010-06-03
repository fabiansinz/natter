Product of Exponential Power Distributions
==========================================

This distribution class implements the product of exponential power
distribution. The joint distribution is factorial with each marginal
being an exponential power distribution

.. math::

   p(\boldsymbol{x}) = \prod_{i=1}^n p(x_i|p_i, s_i)
   
   p(x_i) = \frac{p_i}{2s_i^\frac{1}{p_i} \Gamma\left[\frac{1}{p_i}\right]} \exp\left(\frac{|x_i|^p_i}{s_i}\right)

.. autoclass:: natter.Distributions.ProductOfExponentialPowerDistributions
   :members: sample, loglik, pdf, dldx, estimate, all, copy

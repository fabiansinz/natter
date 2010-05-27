Univariate Mixture of Gaussians
===============================

The mixture of Gaussians has the form

.. math::

   p(\boldsymbol{x}) = \sum_{k=1}^K \pi_k p(x,\boldsymbol \mu_k,
   \boldsymbol \sigma_k)

where :math:`p(x,\boldsymbol \mu_k,\boldsymbol \sigma_k)` is a univariate Gaussian distribution with mean and variance :math:`\boldsymbol \mu_k` and :math:`\boldsymbol \sigma_k`.


.. autoclass:: natter.Distributions.MixtureOfGaussians
   :members: sample, loglik, pdf, estimate, all, copy, cdf

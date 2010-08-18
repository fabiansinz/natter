Normal-Inverse-Gaussian (NIG) distribution
==========================================

The distribution object implements the normal inverse gaussian distribution with
the parametrization 

.. math::

   p(x) = \int \mathcal{N}(x|\mu + \beta z \Gamma, z\Gamma) IG(z,\delta^2,\alpha^2 - \beta^\top \Gamma\beta) d z

  

.. autoclass:: natter.Distributions.NIG
   :members: sample, loglik, pdf, cdf, ppf, estimate, all

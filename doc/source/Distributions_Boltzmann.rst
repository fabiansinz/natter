Boltzmann distribution
=======================

The distribution object implements the Boltzmann distribution with
the parametrization 

.. math::

    p(b) = \frac{1}{Z}\exp( b^\top (L + L^\top) b  + h^\top b) , \mbox{where}\\
    Z   = \sum_b \exp( b^\top (L +L^\top) b + h^\top b )

Where L is a lower triangular matrix with zero diagonal and h is a bias term. b_i \in {0,1}

.. autoclass:: natter.Distributions.Boltzmann
   :members: sample, loglik, pdf, cdf, estimate, estimatePartitionFunction, all

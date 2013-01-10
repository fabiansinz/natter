ChiP distribution
=================

Univariate distribution which is the radial distribution of a
p-generalized Normal.

.. math::

   p(x) =  \frac{p\cdot x^{n-1}}{\Gamma\left(\frac{n}{p}\right)\,s^\frac{n}{p}} \exp{\left(-\frac{x^p}{s}\right)}



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.ChiP
   :members: sample, loglik, pdf, estimate, all, copy, parameters

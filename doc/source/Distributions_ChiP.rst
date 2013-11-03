ChiP distribution
=================

Univariate distribution which is the radial distribution of a
p-generalized Normal.

.. math::

   p(x) =  \frac{p\cdot x^{n-1}}{\Gamma\left(\frac{n}{p}\right)\,s^\frac{n}{p}} \exp{\left(-\frac{x^p}{s}\right)}

where :math:`n\in \mathbb N` refers to the dimensionality of the p-generalized Normal, :math:`p > 0` refers to the
contour line of the distribution, and :math:`s>0` is the scale parameter.


.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.ChiP
   :members:

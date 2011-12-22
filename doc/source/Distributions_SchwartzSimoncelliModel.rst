Schwartz Simoncelli Model
=========================

The Schwartz Simoncelli model is not a true joint probability
distribution. Instead, it is the product of n 1-dimensional
conditional distributions (Gaussians) for which the variance depends
on the (n-1) other values.


.. math::

   p(\boldsymbol{y}) = \prod_{i=1}^n p(y_i|\boldsymbol y_{\hat i})

with 

.. math::
   p(y_i|\boldsymbol y_{\hat i}) = \frac{1}{\sqrt{2\pi \left(\sum_j  w_{ij} y_j^2 + \sigma^2 \right)}}\, \exp\left({ -\frac{y_i^2}{2 \left(\sum_j w_{ij} y_j^2 + \sigma^2 \right)}}\right)



.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. autoclass:: natter.Distributions.SchwartzSimoncelliModel
   :members: loglik, primary2array, array2primary, dldtheta,estimate

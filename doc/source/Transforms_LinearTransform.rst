The LinearTransform Class
=========================

The LinearTransform class respresents linear transformation of
data. Internally, the linear transformation is represented as a matrix
stored in the field *W*. The multiplication operator (among others) of
the LinearTransform class is overloaded, such that
LinearTransform*Data yields Data and LinearTransform*LinearTransform
yields a LinearTransform object again.

IMPORTANT: Internally, all matrices are represented as numpy.array. So
the initializing matrices W should also be arrays and not numpy.matrix.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>
.. moduleauthor:: Philipp Lies <phil@bethgelab.org>
.. autoclass:: natter.Transforms.LinearTransform
   :members:   plotBasis, plotFilters, __invert__, stack, transpose, T, apply,det,logDetJacobian,__getitem__,__str__,getHistory,copy,addToHistory,getOptimalOrientationAndFrequency, shape, html, min, max, __iadd__, __add__, __isub__, __sub__, __idiv__, __div__, __imult__, __mult__, reshape, inv, gratingProjection, fitGauss2Grating


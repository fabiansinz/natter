The LinearTransform1D Class
=========================

The LinearTransform1D class respresents linear transformation of
data. Internally, the linear transformation is represented as a matrix
stored in the field *W*. The multiplication operator (among others) of
the LinearTransform class is overloaded, such that
LinearTransform*Data yields Data and LinearTransform*LinearTransform
yields a LinearTransform object again. The data on which the linear
transform is applies is assumed to be 1D in contrast to the flattened
2D data of the LinearTransform class.

IMPORTANT: Internally, all matrices are represented as numpy.array. So
the initializing matrices W should also be arrays and not numpy.matrix.

.. autoclass:: natter.Transforms.LinearTransform1D
   :members:   plotBasis, plotFilters



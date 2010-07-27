The NonlinearTransform Class
============================

The LinearTransform class respresents linear transformation of
data. Internally, the linear transformation is represented as a matrix
stored in the field *W*. The multiplication operator (among others) of
the LinearTransform class is overloaded, such that
LinearTransform*Data yields Data and LinearTransform*LinearTransform
yields a LinearTransform object again.


.. autoclass:: natter.Transforms.NonlinearTransform
   :members:   




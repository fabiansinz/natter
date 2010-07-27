The NonlinearTransform Class
============================

The NonlinearTransform class represents all non-linear transformations
of data. Like LinearTransform objects it has an overloaded
multiplication operation which allows it to be multiplied (applied to
the results of) LinearTransform and Data objects.

.. autoclass:: natter.Transforms.NonlinearTransform
   :members:   apply, logDetJacobian,  __call__,  __str__, copy,addToHistory




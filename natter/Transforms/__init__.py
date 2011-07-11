"""
The Transforms module contains classes for transforming data in Data
objects. The largest part are linear transforms, but also a class for
nonlinear transforms is provided. Both classes also have associated
factories.

.. toctree::
   :maxdepth: 2
   
   Transforms_LinearTransform.rst
   Transforms_LinearTransform1D.rst
   Transforms_LinearTransformFactory.rst
   Transforms_NonlinearTransform.rst
   Transforms_NonlinearTransformFactory.rst
   Transforms_Transform.rst

"""



from Transform import Transform,load
from LinearTransform import LinearTransform
from NonlinearTransform import NonlinearTransform
import LinearTransformFactory
import LinearTransformUtils
import NonlinearTransformFactory
from LinearTransform1D import LinearTransform1D

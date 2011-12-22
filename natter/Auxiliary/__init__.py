"""

The Auxiliary module includes several auxiliary functions for the natter. It contains the following submodules.

.. toctree::
   :maxdepth: 2
   
   Auxiliary_Decorators.rst
   Auxiliary_Entropy.rst
   Auxiliary_ImageUtils.rst
   Auxiliary_LpNestedFunction.rst
   Auxiliary_Numerics.rst
   Auxiliary_Optimization.rst
   Auxiliary_Plotting.rst
   Auxiliary_Utils.rst
"""

from Utils import save, debug , profileFunction,  mnorm
import Errors 
import Optimization
import Plotting
import Numerics
import Entropy
import ImageUtils
import Decorators
from LpNestedFunction import LpNestedFunction


def prettyPrintDict(d):
    s = ""
    for k ,v in d.items():
        s += str(k) + ": " + str(v) + "\n"
    return s
        
        

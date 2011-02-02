"""

The Distributions module stores all distributions the *natter*
implements. At the moment it provides the following distributions.

.. toctree::
   :maxdepth: 2
   
   Distributions_CompleteLinearModel.rst
   Distributions_Dirichlet.rst
   Distributions_Distribution.rst
   Distributions_ExponentialPower.rst
   Distributions_Gamma.rst
   Distributions_GammaP.rst
   Distributions_Gaussian.rst
   Distributions_ISA.rst
   Distributions_Kumaraswamy.rst
   Distributions_LogNormal.rst
   Distributions_LpNestedSymmetric.rst
   Distributions_LpSphericallySymmetric.rst
   Distributions_MixtureOfGaussians.rst
   Distributions_MixtureOfLogNormals.rst
   Distributions_ProductOfExponentialPowerDistributions.rst
   Distributions_SchwartzSimoncelliModel.rst
   Distributions_TruncatedGaussian.rst
"""


from Distribution import Distribution
from Gamma import Gamma
from GammaP import GammaP
from Dirichlet import Dirichlet
from MixtureOfGaussians import MixtureOfGaussians
from MixtureOfLogNormals import MixtureOfLogNormals
from LpSphericallySymmetric import LpSphericallySymmetric
from LpGeneralizedNormal import LpGeneralizedNormal
from CompleteLinearModel import CompleteLinearModel
from LpNestedSymmetric import LpNestedSymmetric
from Gaussian import Gaussian
from Gaussian import Gaussian as MultivariateNormal
from ExponentialPower import ExponentialPower
from LogNormal import LogNormal
from Distribution import load
from ProductOfExponentialPowerDistributions import ProductOfExponentialPowerDistributions
from EllipticallyContourGamma import EllipticallyContourGamma
from SchwartzSimoncelliModel import SchwartzSimoncelliModel
from ISA import ISA
from Kumaraswamy import Kumaraswamy
from TruncatedGaussian import TruncatedGaussian
from TruncatedGaussian import TruncatedGaussian as TruncatedNormal
# from TruncatedSkewedGaussian import TruncatedSkewedGaussian
# from TruncatedSkewedGaussian import TruncatedSkewedNormal

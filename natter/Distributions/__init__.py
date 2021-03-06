"""

The Distributions module stores all distributions the *natter*
implements. At the moment it provides the following distributions.

.. toctree::
   :maxdepth: 2

   Distributions_ChiP.rst
   Distributions_CompleteLinearModel.rst
   Distributions_Delta.rst
   Distributions_Dirichlet.rst
   Distributions_Distribution.rst
   Distributions_ExponentialPower.rst
   Distributions_FiniteMixtureDistribution.rst
   Distributions_Gamma.rst
   Distributions_GammaP.rst
   Distributions_Gaussian.rst
   Distributions_ISA.rst
   Distributions_Histogram.rst
   Distributions_Kumaraswamy.rst
   Distributions_LogLogistic.rst
   Distributions_LogNormal.rst
   Distributions_LpNestedSymmetric.rst
   Distributions_LpSphericallySymmetric.rst
   Distributions_MixtureOfGaussians.rst
   Distributions_MixtureOfLogNormals.rst
   Distributions_NakaRushton.rst
   Distributions_PCauchy.rst
   Distributions_ProductOfExponentialPowerDistributions.rst
   Distributions_SkewedGaussian.rst
   Distributions_SkewedNormal.rst
   Distributions_Truncated.rst
   Distributions_TruncatedExponentialPower.rst
   Distributions_TruncatedGaussian.rst
   Distributions_TruncatedNormal.rst
   Distributions_Transformed.rst
   Distributions_Uniform.rst
"""


from Distribution import Distribution
from Beta import Beta
from ChiP import ChiP
from Gamma import Gamma
from GammaP import GammaP
from Dirichlet import Dirichlet
from MixtureOfGaussians import MixtureOfGaussians
from MixtureOfLogNormals import MixtureOfLogNormals
from LpSphericallySymmetric import LpSphericallySymmetric
from LpGeneralizedNormal import LpGeneralizedNormal
from LogLogistic import LogLogistic
from CompleteLinearModel import CompleteLinearModel
from LpNestedSymmetric import LpNestedSymmetric
from ExponentialPower import ExponentialPower
from Gaussian import Gaussian
from Gaussian import Gaussian as MultivariateNormal
from Histogram import Histogram
from LogNormal import LogNormal
from NakaRushton import NakaRushton
from ProductOfExponentialPowerDistributions import ProductOfExponentialPowerDistributions
from EllipticallyContourGamma import EllipticallyContourGamma
from ISA import ISA
from Kumaraswamy import Kumaraswamy
from TruncatedGaussian import TruncatedGaussian
from TruncatedGaussian import TruncatedGaussian as TruncatedNormal
from SkewedGaussian import SkewedGaussian
from SkewedGaussian import SkewedGaussian as SkewedNormal
from TruncatedExponentialPower import TruncatedExponentialPower
from FiniteMixtureDistribution import FiniteMixtureDistribution
from Truncated import Truncated
from Transformed import Transformed
from TModelRadial import TModelRadial
from Uniform import Uniform
from Delta import Delta
from PCauchy import PCauchy

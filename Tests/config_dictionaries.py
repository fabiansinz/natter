"""
This file contains a list of dictionaries of all distributions which
should be tested with the generic test suite.
"""
from natter.Distributions import *
from natter.Auxiliary import LpNestedFunction
import numpy as np

OODLES = int(1e6)

distributions_to_test = []


# # # full example with all optional arguments specified
distributions_to_test.append({'dist'      :Gaussian({'n':1,'sigma':np.eye(1)}), # distribution to test
                              'nsamples'  :10000, # number of samples to use for importance sampling testing
                              'tolerance' :1e-01, # allowed deviation from 1  for the normalisation constant
                              'support'   : (-np.inf,np.inf), # support of the distribution, can either be -inf,inf ; 0,inf or bounded
                              'proposal_high'  : Gaussian({'n':1,'sigma':np.eye(1)*5}), # proposal distribution for importance sampling with low variance (optional)
                              'proposal_low'  : Gaussian({'n':1,'sigma':np.eye(1)*0.1}) # proposal with high variance, also for importance sampling
                              }) 


distributions_to_test.append({'dist'      :Gaussian({'n':1,'sigma':np.eye(1)})}) 

# # minimal example
distributions_to_test.append({'dist'      :Gaussian })


distributions_to_test.append({'dist'      :ChiP,
                              'support'   : (0,np.inf),
                              'nsamples'  : OODLES,
                              'tolerance':1e-01})


distributions_to_test.append({'dist'      :ExponentialPower,
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :FiniteMixtureDistribution,
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :Gamma,
                              'nsamples': 5*OODLES,
                              'support': (0,np.inf),
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :GammaP,
                              'nsamples':1e07,    #we need oodles and oodles of samples
                              'support': (0,np.inf),
                              'tolerance':1e-01,
                              'proposal_high'  : Gamma(s=10.), # proposal distribution for importance sampling with low variance (optional)
                              'proposal_low'  : Gamma(s=1.) # proposal with high variance, also for importance sampling
                             })


distributions_to_test.append({'dist'      :ISA,
                              'tolerance':1e-01                            
                              })

distributions_to_test.append({'dist'      :Kumaraswamy,
                              'support': (0,1),
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :LogLogistic,
                              'support': (0,np.inf),
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :LogNormal,
                              'support': (0,np.inf),
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :LpGeneralizedNormal,
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :LpNestedSymmetric(n=3, rp=Gamma(),f=LpNestedFunction('(0,0,(1,1:3))',[1.0,1.5])),
                              'tolerance':1e-01,
                              'nsamples': 3*OODLES,
                              'proposal_high': Gaussian(n=3,sigma=np.eye(3)*10.0),
                              'proposal_low': Gaussian(n=3,sigma=np.eye(3)*2.0)
                              })

distributions_to_test.append({'dist'      :LpSphericallySymmetric,
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :MixtureOfGaussians,
                              'nsamples' :10*OODLES,
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :MixtureOfLogNormals,
                              'support': (0,np.inf),
                              'nsamples' :5*OODLES,
                              'proposal_high'  : Gamma(s=25.), # proposal distribution for importance sampling with low variance (optional)
                              'proposal_low'  : Gamma(s=1.), # proposal with high variance, also for importance sampling

                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :NakaRushton,
                              'support': (0,np.inf),
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :PCauchy,
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :ProductOfExponentialPowerDistributions,
                              'tolerance':1e-01})


distributions_to_test.append({'dist'      :SkewedGaussian,
                              'tolerance':1e-01})


distributions_to_test.append({'dist'      :TruncatedExponentialPower(a=-5,b=5),
                              'support':(-5,5),
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :TruncatedGaussian(a=0,b=1),
                              'support':(0,1),
                              'tolerance':1e-01})

distributions_to_test.append({'dist'      :Uniform,
                              'support':(0,1),
                              'tolerance':1e-01})


distributions_to_test.append({'dist'      :EllipticallyContourGamma,
                              'nsamples': 10*OODLES, #we need an awful lot of samples
                              'proposal_high'  : Gaussian({'n':2,'sigma':np.eye(2)*30}), # proposal distribution for importance sampling with low variance (optional)
                              'proposal_low'  : Gaussian({'n':2,'sigma':np.eye(2)*2}), # proposal with high variance, also for importance sampling
                              'tolerance':2e-01})

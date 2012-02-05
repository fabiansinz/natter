from natter.Distributions import *
import numpy as np
distributions_to_test = []

distributions_to_test.append({'dist'      :Gaussian({'n':1,'sigma':np.eye(1)}),
                              'nsamples'  :10000,
                              'tolerance' :1e-01,
                              'support'   : (-np.inf,np.inf),
                              'proposal_high'  : Gaussian({'n':1,'sigma':np.eye(1)*5}),
                              'proposal_low'  : Gaussian({'n':1,'sigma':np.eye(1)*0.1}),
                              'basedist': Gaussian})


distributions_to_test.append({'dist'      :Gaussian({'n':1,'sigma':np.eye(1)})}) 

distributions_to_test.append({'dist'      :Gaussian })


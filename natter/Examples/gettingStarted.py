
from numpy.random import randn
from natter import DataModule
from numpy import dot
from natter import Distributions

S = randn(3,3)
S = dot(S,S.T)
dat = DataModule.DataSampler.gauss(3,10000,sigma=S)

print dat

p = Distributions.Gaussian({'n':3})

print p

p.estimate(dat)
print p

dat2 = p.sample(50000)
print dat2

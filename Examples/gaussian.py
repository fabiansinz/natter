import Distribution
from numpy.random import randn
from numpy import dot

if __name__=="__main__":
    sigma = randn(3,3)
    sigma = dot(sigma,sigma.transpose())
    p = Distribution.Gaussian({'n':3,'sigma':sigma})
    print p
    dat = p.sample(10000)
    print dat
    print dat.cov()

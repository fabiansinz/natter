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
    dat = p.sample(10)
    print p.loglik(dat)
    print p.pdf(dat)


#     print p.primary2array()
#     dummy = randn(9)
#     print dummy
#     p.array2primary(dummy)
#     print p

    p.primary = ['mu','sigma']
    print p.dldtheta(dat)


    df = p.dldtheta(dat)
    df2 = 0.0*df
    h = 1e-6
    pa0 = p.primary2array()

    for k in xrange(len(pa0)):
        pa = pa0.copy()
        pa[k] = pa[k]+ h
        df2[k] = sum(p(dat,pa) - p(dat,pa0))/h

    print df
    print df2
    print df-df2
#     p2 = Distribution.Gaussian({'n':1})
#     dat = p2.sample(100000)
#     p2.histogram(dat)

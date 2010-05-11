import Distributions
import numpy as np

if __name__=="__main__":
    a1 = np.random.rand(3,2)
    a2 = np.random.rand(3,2)
    p = Distributions.MixtureOfDirichlet({'K':2,'alpha':a1})
    p2 = Distributions.MixtureOfDirichlet({'K':2,'alpha':a2})
    print p
    print p2
    print p2.param['alpha']-p.param['alpha']
    print p2.param['pi']-p.param['pi']
    dat = p.sample(10000)
    
    p2.estimate(dat)
    print p2.param['alpha']-p.param['alpha']
    print p2.param['pi']-p.param['pi']

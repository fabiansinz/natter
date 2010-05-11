import numpy as np
import Transform
import Distributions

if __name__=="__main__":
    p = Distributions.GammaP({'u':np.random.rand()*5.0,'s':np.random.rand()*10.0,'p':np.random.rand()*1.5+.5})
    print p
    dat = p.sample(10000)
    p.histogram(dat)

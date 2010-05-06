import numpy as np
import Filter
import Distribution
import Data

if __name__=="__main__":
    # source distribution
    psource = Distribution.LpSphericallySymmetric({'p':1.0})
    # create Filter
    F = Filter.FilterFactory.RadialFactorization(psource)
    # sample data from source distribution
    dat = psource.sample(100000)
    
    # apply filter to data
    dat2 = F*dat

    print dat
    print dat2
    print F
    

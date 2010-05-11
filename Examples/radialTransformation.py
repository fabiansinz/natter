import numpy as np
import Transform
import Distributions
from DataModule import Data

if __name__=="__main__":
    p = Distributions.LpSphericallySymmetric({'p':1.0})
    # source distribution
    psource = Distributions.LpSphericallySymmetric({'p':1.0})
    # target distribution
    ptarget = Distributions.LpSphericallySymmetric({'p':1.0,'rp':Distributions.Gamma({'u':np.random.rand()*3.0,'s':np.random.rand()*2.0})})
    # create Transform
    F = Transform.TransformFactory.RadialTransformation(psource,ptarget)
    # sample data from source distribution
    dat = psource.sample(10)
    
    # apply filter to data
    dat2 = F*dat
    logDetJ =  F.logDetJacobian(dat)
    logDetJ2 = 0*logDetJ
    n = dat.size(0)

    h = 1e-6

    tmp = Data(dat.X.copy())
    tmp.X[0,:] += h
    W1 = ((F*tmp).X-dat2.X)/h

    tmp = Data(dat.X.copy())
    tmp.X[1,:] += h
    W2 = ((F*tmp).X-dat2.X)/h
            
    for i in range(dat.size(1)):
        logDetJ2[i] = np.log(np.abs(W1[0,i]*W2[1,i] - W1[1,i]*W2[0,i]))

    # compare all of new and old distribution corrected with the logdeterminant
    print psource.all(dat) 
    print ptarget.all(dat2) - np.mean(logDetJ2)/n/np.log(2)
    print ptarget.all(dat2) - np.mean(logDetJ)/n/np.log(2)
    

    # estimate transformed data to check that transformation is right
    p.estimate(dat2)
    # plot the original and the fitted target distribution
    print p
    print ptarget

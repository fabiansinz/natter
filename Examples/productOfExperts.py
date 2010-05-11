import Distributions
from DataModule import Data
import Transforms
import numpy as np

if __name__=="__main__":
    dat = Data.DataLoader.load('Examples/dat4x4.dat')
    print dat
    tmp = Transform.TransformFactory.DCnonDC(dat)
    FDC = tmp[0,:]
    FrDC = tmp[1:,:]
    FSYM = Transform.TransformFactory.SYM(FrDC*dat)
    
    p = Distributions.ProductOfExperts({'n':15})
    p.estimate(FSYM*FrDC*dat)
    p.save('4x4PoE')

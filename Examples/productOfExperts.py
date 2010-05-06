import Distribution
import Data
import Filter
import numpy as np

if __name__=="__main__":
    dat = Data.DataLoader.load('Examples/dat4x4.dat')
    print dat
    tmp = Filter.FilterFactory.DCnonDC(dat)
    FDC = tmp[0,:]
    FrDC = tmp[1:,:]
    FSYM = Filter.FilterFactory.SYM(FrDC*dat)
    
    p = Distribution.ProductOfExperts({'n':15})
    p.estimate(FSYM*FrDC*dat)
    p.save('4x4PoE')

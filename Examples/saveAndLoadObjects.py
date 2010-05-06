import Data
import numpy as np
import Distribution
import Auxiliary
import Filter

if __name__=="__main__":
    dat = Data.Data(np.random.randn(4,10),'My Data')
    print dat
    dat.save('TestSave')
    dat2 = Data.DataLoader.load('TestSave.pydat')
    print dat2


    ##############################################
    l = Auxiliary.LpNestedFunction()
    p = Distribution.LpNestedSymmetric({'f':l})
    print p
    p.save('TestSave2')
    p2 = Distribution.load('TestSave2.pydat')
    print p2

    ##############################################
    F = Filter.FilterFactory.oRND(dat2)
    print F
    F.save('TestSave3')
    F2 = Filter.load('TestSave3.pydat')
    print F2

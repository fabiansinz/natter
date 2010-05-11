from DataModule import Data, DataLoader
import numpy as np
import Distributions
import Auxiliary
import Transforms

if __name__=="__main__":
    dat = Data(np.random.randn(4,10),'My Data')
    print dat
    dat.save('TestSave')
    dat2 = DataLoader.load('TestSave.pydat')
    print dat2


    ##############################################
    l = Auxiliary.LpNestedFunction()
    p = Distributions.LpNestedSymmetric({'f':l})
    print p
    p.save('TestSave2')
    p2 = Distributions.load('TestSave2.pydat')
    print p2

    ##############################################
    F = Transform.TransformFactory.oRND(dat2)
    print F
    F.save('TestSave3')
    F2 = Transform.load('TestSave3.pydat')
    print F2

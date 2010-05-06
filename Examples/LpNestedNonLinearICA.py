import Distribution
import Data
import numpy as np
import Auxiliary
import matplotlib as mpl
import Filter
from numpy import linalg

if __name__=='__main__':

#    L = Auxiliary.LpNestedFunction('(0,(1,0:2),2:4)')
#    L = Auxiliary.LpNestedFunction('(0,0:5)')
    L = Auxiliary.LpNestedFunction()
    p = Distribution.LpNestedSymmetric({'f':L})
    dat = p.sample(50000)
    F = Filter.FilterFactory.LpNestedNonLinearICA(p,dat)
    print p
    print F
    dat2 = F*dat

    for k in L.n.keys():
        if L.n[k] == 1:
            i = L.i(k)
            pnew = Distribution.ExponentialPower({'s':p.param['rp'].param['s'], 'p':L.p[L.pdict[k[:-1]]]})
            pnew.histogram(dat2[i,:])
            raw_input()
            

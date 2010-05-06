import Distribution
import numpy as np
import Auxiliary
import matplotlib as mpl
import Filter
from numpy import linalg
import pickle


if __name__=='__main__':


    F = Filter.LinearFilter(np.random.rand(100,100))
    L = Auxiliary.LpNestedFunction('(0,(0,0:20),20:25,(1,25:45),45:50,(1,75:100))')
    print L.p
    print L
    L.plotGraph(F)

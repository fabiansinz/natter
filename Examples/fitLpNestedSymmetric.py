import Auxiliary
import Distributions
import pylab as pl
import  numpy as np
if __name__=="__main__":
    # create function object
    L =Auxiliary.LpNestedFunction('(0,0,(1,1:3),3)')
    L.p = np.random.rand(2)*2.0
    L.lb = 0.0*L.p
    L.ub = 0.0*L.p+2.0
    d = Distributions.LpNestedSymmetric({'f':L,'n':L.n[()]})
    print d
    L2 =Auxiliary.LpNestedFunction('(0,0,(1,1:3),3)')
    L2.p = np.random.rand(2)*2.0
    rd2 = Distributions.Gamma({'u':5*np.random.rand(),'s':10*np.random.rand()})
    # create Distributions object and sample
    d2 = Distributions.LpNestedSymmetric({'f':L2,'n':L2.n[()],'rp':rd2})
    print d2
    dat = d2.sample(10000)
    d.estimate(dat,method="greedy")
    print d
    d.estimate(dat,method="neldermead")
    print d

    
    
    


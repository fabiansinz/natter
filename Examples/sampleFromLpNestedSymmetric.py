import Auxiliary
import Distributions
import pylab as pl
import  numpy as np
if __name__=="__main__":
    # create function object
    L =Auxiliary.LpNestedFunction('(0,0,(1,1:3),3)')
    L.p = np.random.rand(2)*2.0
    print L
    # create Distributions object and sample
    d = Distributions.LpNestedSymmetric({'f':L,'n':L.n[()]})

    plrange = 2*np.array([-1,1,-1,1])
    # plot samples
    dat = d.sample(10000)
    pl.subplot(231)
    pl.plot(dat.X[0,:],dat.X[1,:],marker='.',markersize=1.0,linestyle='None')
    pl.axis(plrange)
    pl.title('Dimensions (0,1)')
    pl.grid(True)
    pl.subplot(232)
    pl.plot(dat.X[0,:],dat.X[2,:],marker='.',markersize=1.0,linestyle='None')
    pl.axis(plrange)
    pl.title('Dimensions (0,2)')
    pl.grid(True)
    pl.subplot(233)
    pl.plot(dat.X[0,:],dat.X[3,:],marker='.',markersize=1.0,linestyle='None')
    pl.axis(plrange)
    pl.title('Dimensions (0,3)')
    pl.grid(True)
    pl.subplot(234)
    pl.plot(dat.X[1,:],dat.X[2,:],marker='.',markersize=1.0,linestyle='None')
    pl.axis(plrange)
    pl.title('Dimensions (1,2)')
    pl.grid(True)
    pl.subplot(235)
    pl.plot(dat.X[1,:],dat.X[3,:],marker='.',markersize=1.0,linestyle='None')
    pl.axis(plrange)
    pl.title('Dimensions (1,3)')
    pl.grid(True)
    pl.subplot(236)
    pl.plot(dat.X[2,:],dat.X[3,:],marker='.',markersize=1.0,linestyle='None')
    pl.title('Dimensions (2,3)')
    pl.axis(plrange)
    pl.grid(True)

    pl.show()
    
    
    


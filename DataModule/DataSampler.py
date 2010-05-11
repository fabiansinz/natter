from numpy import shape, floor, zeros, NaN, isinf, isnan, any, array
from numpy.random import rand
from Data import Data


def img2PatchRand(img, p, N):
    """

    img2PatchRand(img, p, N)

    samples N pXp patches from an numpy array img and returns an
    appropriate Data object. The images are vectorized in
    FORTRAN/MATLAB style.
    
    """

    ny,nx = shape(img)

    x = floor( rand( N, 1) * ( nx - p + 1)) 
    y = floor( rand( N, 1) * ( ny - p + 1))

    p1 = p - 1
  
    X = zeros( ( p*p, N))

    for ii in xrange(N):
        ptch = array([NaN])
        while any( isnan( ptch.flatten())) or any( isinf(ptch.flatten())) or any(ptch.flatten() == 0.0): 
            xi = floor( rand() * ( nx - p))
            yi = floor( rand() * ( ny - p))
            ptch = img[ yi:yi+p1+1, xi:xi+p1+1]
            X[:,ii] = ptch.flatten('F')
  
    name = "%d %dX%d patches" % (N,p,p)
    return Data(X, name)
from __future__ import division
import numpy as np
import mdp
from LinearTransform import LinearTransform
from numpy import linalg
from scipy import special
from Auxiliary import Errors
import Auxiliary
import types

################################ LINEAR FILTERS ########################################


def fastICA(dat,whitened=True):
    sampsize = 1
    if dat.size(1) > 500000:
      sampsize =  500000.0/dat.size(1)
    # g = 'gaus'?
    ICA = mdp.nodes.FastICANode(input_dim=dat.size(0),approach='symm',stabilization=True,sample_size=sampsize,max_it=10000,max_it_fine=1000)
    ICA.whitened = whitened
    ICA.verbose = True
    ICA.train(dat.X.transpose())
    return LinearTransform(ICA.get_projmatrix().transpose(),'fast ICA filter computed on ' + dat.name)
    
    

def wPCA(dat):
    wPCA = mdp.nodes.WhiteningNode(input_dim=dat.size(0),output_dim=dat.size(0))
    wPCA.verbose=True
    wPCA.train(dat.X.transpose())
    return LinearTransform(wPCA.get_projmatrix().transpose(),'Whitening PCA filter computed on ' + dat.name)
    

def DCnonDC(dat):
    n = dat.size(0)
    P = np.eye(n)
    P[:,0] = 1
    (Q,R) = linalg.qr(P)
    return LinearTransform(Q.transpose(),'DC|nonDC filter in ' + str(n) + ' dimensions')

def SYM(dat):
    C = np.cov(dat.X)
    (V,U) = linalg.eig(C)
    V = np.diag( V**(-.5))
    return LinearTransform(np.dot(U,np.dot(V,U.transpose())),'Symmetric whitening filter computed on ' + dat.name)

def oRND(dat):
    return LinearTransform(mdp.utils.random_rot(dat.size(0)),'Random rotation matrix',['sampled from a Haar distribution'])


def stRND(sh):
    return LinearTransform( Auxiliary.Optimization.projectOntoSt(np.random.randn(sh[0],sh[1])),'Random Stiefel Matrix')
    
            
def DCT2(sh):
    cos = np.cos
    pi = np.pi
    sqrt = np.sqrt
    
    if type(sh) == types.TupleType:
        N1,N2 = sh
    else:
        N1 = sh
        N2 = sh
    F = np.zeros((N1*N2,N1*N2))
    x = np.reshape(np.array(range(N1)),(N1,1))
    y = np.reshape(np.array(range(N2)),(1,N2))
    for i in xrange(N1):
        for j in xrange(N2):
            tmp = np.zeros((N1,N2))
            tmp = np.dot(cos(pi/N1 * (x +.5)* i), cos(pi/N2 * (y + .5) * j))
            F[i*N1 + j,:] = tmp.copy().flatten() / sqrt(sum(tmp.flatten()**2))
    return LinearTransform(F, str(N1) + 'X' + str(N2) + ' 2D-DCT orthonormal Basis')
            
    

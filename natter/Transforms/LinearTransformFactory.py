from __future__ import division
import numpy as np
import mdp
from LinearTransform import LinearTransform
from numpy import linalg
from natter import Auxiliary
import types


################################ LINEAR FILTERS ########################################


def fastICA(dat,whitened=True):
    """
    Creates a linear filter that contains the demixing matrix of a
    complete linear ICA, fitted with fast ICA.

    :param dat: Data on which the ICA will be computed.
    :type dat: natter.DataModule.Data
    :param whitened: Indicates whether the data is whitened or not.
    :type whitened: bool
    :returns: A linear filter containing the fast ICA demixing matrix.
    :rtype: natter.Transforms.LinearTransform
    
    
    """
    sampsize = 1
    if dat.size(1) > 500000:
        sampsize =  500000.0/dat.size(1)
    ICA = mdp.nodes.FastICANode(input_dim=dat.size(0),limit=1e-5, fine_gaus=1.0, fine_g='gaus', g='gaus',\
                                mu=1.0,  approach='symm',stabilization=True,sample_size=sampsize,\
                                max_it=1000,max_it_fine=20)
    ICA.whitened = whitened
    ICA.verbose = True
    ICA.train(dat.X.T)
    # refine
    # ICA.g = 'gaus'
    # ICA.train(dat.X.transpose())
    return LinearTransform(Auxiliary.Optimization.projectOntoSt(ICA.get_projmatrix(False)),'fast ICA filter computed on ' + dat.name)
    
    

def wPCA(dat,m=None):
    """
    Creates a linear filter that projects the data onto the principal
    components and scales it by the inverse standard deviation along
    those directions (i.e. whitens the data).

    :param dat: Data on which the whitening will be computed.
    :type dat: natter.DataModule.Data
    :param m:  Number of components to be kept.
    :type m: int
    :returns: A linear filter containing the whitening matrix.
    :rtype: natter.Transforms.LinearTransform
    
    
    """
    if m==None:
        m = dat.size(0)
    wPCA = mdp.nodes.WhiteningNode(input_dim=dat.size(0),output_dim=m)
    wPCA.verbose=True
    wPCA.train(dat.X.transpose())
    return LinearTransform(wPCA.get_projmatrix().transpose(),'Whitening PCA filter computed on ' + dat.name)
    
def DCAC(dat):
    """
    Creates a linear filter whose first row corresponds to the DC
    component of the data. The other rows correspond to the AC
    components. The matrix is chosen to be orthonormal.

    :param dat: Data for which the DC/AC components will be computed.
    :type dat: natter.DataModule.Data
    :returns: A linear filter containing the DC/AC matrix.
    :rtype: natter.Transforms.LinearTransform
    
    
    """
    return DCnonDC(dat)

def DCnonDC(dat):
    """
    Equivalent to DCAC.
    
    :param dat: Data for which the DC/AC components will be computed.
    :type dat: natter.DataModule.Data
    :returns: A linear filter containing the DC/AC matrix.
    :rtype: natter.Transforms.LinearTransform
    
    
    """
    n = dat.size(0)
    P = np.eye(n)
    P[:,0] = 1
    (Q,R) = linalg.qr(P)
    return LinearTransform(Q.transpose(),'DC|nonDC filter in ' + str(n) + ' dimensions')

def SYM(dat):
    """
    Creates a linear filter that whitens the data with
    symmetric/zero-phase whitening.

    :param dat: Data on which the whitening will be computed.
    :type dat: natter.DataModule.Data
    :returns: A linear filter containing the whitening matrix.
    :rtype: natter.Transforms.LinearTransform
    
    
    """
    
    C = np.cov(dat.X)
    (V,U) = linalg.eig(C)
    V = np.diag( V**(-.5))
    return LinearTransform(np.dot(U,np.dot(V,U.transpose())),'Symmetric whitening filter computed on ' + dat.name)

def oRND(dat):
    """
    Creates a random orthonormal matrix.

    :param dat: Data that determines the correct dimensions of the random orthonormal matrix.
    :type dat: natter.DataModule.Data
    :returns: A linear filter containing the matrix.
    :rtype: natter.Transforms.LinearTransform
    
    """
    return LinearTransform(mdp.utils.random_rot(dat.size(0)),'Random rotation matrix',['sampled from a Haar distribution'])


def stRND(sh):
    """
    Creates a random matrix from the Stiefel manifold. 

    :param sh: Tuple that determines the dimensions of the matrix.
    :type dat: tuple of int
    :returns: A linear filter containing the matrix.
    :rtype: natter.Transforms.LinearTransform
    
    """
    return LinearTransform( Auxiliary.Optimization.projectOntoSt(np.random.randn(sh[0],sh[1])),'Random Stiefel Matrix')
    
            
def DCT2(sh):
    """
    Creates a 2D DCT basis for sh=(N1,N2) image patches.

    :param sh: Tuple that determines the image patch size.
    :type sh: tuple of int
    :returns: A linear filter containing the DCT basis.
    :rtype: natter.Transforms.LinearTransform
    
    """
    
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


from __future__ import division
import numpy as np
import mdp
from LinearTransform import LinearTransform
from numpy import linalg
from natter import Auxiliary
from natter import DataModule
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
    :type sh: tuple of int
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

def DFT(sh):
    """
    Creates a 1D DFT basis for sh X 1 image stripes.

    :param sh: Int that determines the image stripe length.
    :type sh: int
    :returns: A linear filter containing the DFT basis, DC filter as first component.
    :rtype: natter.Transforms.LinearTransform
    
    """
    if not type(sh) == types.IntType:
        raise TypeError('DFT requires single integer as parameter, not %s'%(type(sh)))

    A = np.eye(sh)
    Ufft = np.fft.ifft(A,axis=0)
    Ufft = Ufft.real + Ufft.imag
    Ufft /= np.linalg.norm(Ufft[:,0])

    return LinearTransform(Ufft.T, str(sh) + 'X1' + ' 1D-DFT orthonormal Basis')

def DFT2(sh):
    """
    Creates a 2D DFT basis for sh X sh image patches.

    :param sh: Int that determines the image patch size.
    :type sh: tuple of int
    :returns: A linear filter containing the DFT basis, DC filter as first component.
    :rtype: natter.Transforms.LinearTransform
    
    """
    if not type(sh) == types.IntType:
        raise TypeError('DFT2 requires single integer as parameter, not %s'%(type(sh)))
    
    A = np.eye(sh**2,sh**2).reshape(sh,sh,sh**2)
    Ufft = np.fft.ifft2(A,axes=(0,1)).reshape(sh**2,sh**2)
    Ufft = Ufft.real + Ufft.imag
    Ufft /= np.linalg.norm(Ufft[:,0])

    return LinearTransform(Ufft.T, str(sh) + 'X' + str(sh) + ' 2D-DFT orthonormal Basis')

def DCACQuadraturePairs1D(sh):
    """
    Creates a 1D DFT basis for sh X 1 image stripes where the filters with identical spatial
    frequency and orientation are paired up, such that filters 1&2, 3&4, ... are quadrature
    pairs. Filter 0 is the DC component.
    Only implemented for uneven dimensional data.

    :param sh: Int that determines the image stripe length.
    :type sh: int
    :returns: A linear filter containing the DFT basis, DC filter as first component.
    :rtype: natter.Transforms.LinearTransform
    
    """
    if not type(sh) == types.IntType:
        raise TypeError('DCACQuadraturePairs1D requires single integer as parameter, not %s'%(type(sh)))
    if np.mod(sh,2) == 0:
        raise NotImplementedError('Quadrature pairs are only implemented for uneven dimensional data.')
    num_quadrature_pairs = (sh-1)//2
    ind = np.hstack(((0,), \
                     np.vstack((np.arange(1,num_quadrature_pairs+1),\
                                np.arange(sh-1,sh-num_quadrature_pairs-1,-1))).T.flatten()))
    F = DFT(sh)
    F = F[ind,:]
    F.addToHistory('Rearranged filters into quadrature pairs.')
    return F

def DCACQuadraturePairs2D(sh):
    """
    Creates a 2D DFT basis for sh X sh image patches where the filters with identical spatial
    frequency and orientation are paired up, such that filters 1&2, 3&4, ... are quadrature
    pairs. Filter 0 is the DC component.
    Only implemented for uneven dimensional data.

    :param sh: Int that determines the image patch size.
    :type sh: tuple of int
    :returns: A linear filter containing the DFT basis, DC filter as first component.
    :rtype: natter.Transforms.LinearTransform
    
    """
    if not type(sh) == types.IntType:
        raise TypeError('DCACQuadraturePairs2D requires single integer as parameter, not %s'%(type(sh)))
    if np.mod(sh,2) == 0:
        raise NotImplementedError('Quadrature pairs are only implemented for uneven dimensional data.')
    
    
    num_quadrature_pairs = (sh-1)//2
    compnum = (sh**2-1)//2
    Aorder = np.zeros((sh,sh))
    Aorder[0,1:num_quadrature_pairs+1] = np.arange(1,num_quadrature_pairs+1)
    Aorder[0,num_quadrature_pairs+1:] = np.arange(num_quadrature_pairs, 0, -1)
    Aorder[1:num_quadrature_pairs+1,0] = np.arange(num_quadrature_pairs+1, 2*num_quadrature_pairs+1)
    Aorder[num_quadrature_pairs+1:,0] = np.arange(2*num_quadrature_pairs, num_quadrature_pairs, -1)
    Aorder[1:num_quadrature_pairs+1,1:] = np.arange(2*num_quadrature_pairs+1, compnum+1).reshape(num_quadrature_pairs, sh-1)
    Aorder[num_quadrature_pairs+1:,1:] = np.arange(compnum, 2*num_quadrature_pairs, -1).reshape(num_quadrature_pairs, sh-1)    
    ind = np.argsort(Aorder.flatten())
    
    F = DFT2(sh)
    F = F[ind,:]
    F.addToHistory('Rearranged filters into quadrature pairs.')
    return F

def SubspaceEnergyWhitening(dat, hasDC=True):
    """
    Equalizes the energy in the subspaces of the quadrature pair filters. The filter
    assumes that the first dimension contains the DC component and all following
    dimensions are paired up into quadrature pairs.

    :param dat: Data on which the whitening will be computed.
    :type dat: natter.DataModule.Data
    :param hasDC: Flag indicating whether dat still contains DC component
    :type hasDC: bool
    :returns: A linear filter containing the whitening matrix.
    :rtype: natter.Transforms.LinearTransform  
    
    """
    var = dat.X.var(1)
    if hasDC:
        if np.mod(dat.dim(),2) == 0:
            raise NotImplementedError('Subspace energy whitening with DC component is only implemented for uneven dimensional data.')
        invDCvar = 1/var[0]
        ACvar = var[1:]
    else:
        if np.mod(dat.dim(),2) == 1:
            raise NotImplementedError('Subspace energy whitening without DC component is only implemented for even dimensional data.')
        invDCvar = ()
        ACvar = var
    
    invACvar = 1/np.repeat(ACvar.reshape(ACvar.size//2,2).sum(1)/2,2)
    W = np.diag(np.hstack((invDCvar, invACvar)))
    F = LinearTransform(W, 'Quadrature pair subspace energy equalization filter computed on ' + dat.name )
    return F


def SSA( dat, *args, **kwargs ):
    """
    Creates a linear filter by applying 2D subspace slowness analysis.

    :param dat: Data set with sequence
    :type dat: natter.DataModule.Data
    :returns: A linear filter containing the SSA filters
    :rtype: natter.Transforms.LinearTransform
    """
    functionValues = kwargs.pop('functionValues', None)
        
    SSA = mdp.nodes.SSANode(input_dim=dat.size(0), **kwargs)
    SSA.train(dat.X.T)
    SSA.stop_training()

    if functionValues:
        return LinearTransform(SSA.U,'2D SSA filter computed on ' + dat.name), SSA.functionValues
    else:
        return LinearTransform(SSA.U,'2D SSA filter computed on ' + dat.name)

def mdpWrapper( dat, nodename, *args, **kwargs ):
    """
    *EXPERIMENTAL WRAPPER - DO NOT USE IF YOU DON'T KNOW WHAT YOU DO*
    Creates a linear filter by training a mdp node

    :param dat: Data set with sequence
    :type dat: natter.DataModule.Data
    :param nodename: name of the mdp node (without "Node"), case sensitive!
    :type nodename: String
    :returns: A linear filter containing the SSA filters
    :rtype: natter.Transforms.LinearTransform
    """
    functionValues = kwargs.pop('functionValues', None)

    if hasattr(mdp.nodes, nodename+'Node'):
        node = getattr(mdp.nodes, nodename+'Node')(input_dim=dat.size(0), **kwargs)
    elif hasattr(mdp.nodes, nodename):
        node = getattr(mdp.nodes, nodename)(input_dim=dat.size(0), **kwargs)
    else:
        print "Couldn't find the node '%s' you specified. Start guessing."%(nodename)
        nodelist = dir(mdp.nodes)
        nodelist_lower = [str.lower(attr) for attr in nodelist]
        if nodelist_lower.count(str.lower(nodename)) > 0:
            print "Matching entries found case insensitive: %d"%(nodelist_lower.count(str.lower(nodename)))
            attr = nodelist[nodelist_lower.index(str.lower(nodename))]
            print "Taking "+attr
            node = getattr(mdp.nodes, attr)(input_dim=dat.size(0), **kwargs)
        elif nodelist_lower.count(str.lower(nodename)+'node') > 0:
            print "Matching entries found case insensitive: %d"%(nodelist_lower.count(str.lower(nodename)+'node'))
            attr = nodelist[nodelist_lower.index(str.lower(nodename)+'node')]
            print "Taking "+attr
            node = getattr(mdp.nodes, attr)(input_dim=dat.size(0), **kwargs)
        else:
            raise ValueError("Couldn't find any matching node. Try updating mdp or check your input '%s'"%(nodename))
                
    node.train(dat.X.T)
    node.stop_training()

    if functionValues:
        return LinearTransform(node.U,'2D %s filter computed on '%(nodename) + dat.name), SSA.functionValues
    else:
        return LinearTransform(node.U,'2D %s filter computed on '%(nodename) + dat.name)

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
                                mu=1.0,  approach='symm',sample_size=sampsize,\
                                max_it=1000,max_it_fine=20)
    ICA.whitened = whitened
    ICA.verbose = True
    ICA.train(dat.X.T)
    # refine
    # ICA.g = 'gaus'
    # ICA.train(dat.X.transpose())
    return LinearTransform(Auxiliary.Optimization._projectOntoSt(ICA.get_projmatrix(False)),'fast ICA filter computed on ' + dat.name)



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

    return LinearTransform( Auxiliary.Optimization._projectOntoSt(np.random.randn(sh[0],sh[1])),'Random Stiefel Matrix')


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

def DCACQuadraturePairs2D(patch_size, num_quadrature_pairs=None):
    """
    Creates a 2D DFT basis for patch_size X patch_size image patches where the filters with identical spatial
    frequency and orientation are paired up, such that filters 1&2, 3&4, ... are quadrature
    pairs. Filter 0 is the DC component. If num_quadrature_pairs is not None then the
    DC component plus the #num_quadrature_pairs lowest quadrature pairs are returned.
    Note that for even patch_size there is no set of quadrature pairs with full rank.

    :param patch_size: Int that determines the image patch size.
    :type patch_size: int
    :param num_quadrature_pairs: Number of quadrature pairs to return
    :type num_quadrature_pairs: int
    :returns: A linear filter containing rearranged DFT basis, DC filter as first component.
    :rtype: natter.Transforms.LinearTransform
    """

    if not type(patch_size) == types.IntType:
        raise TypeError('DCACQuadraturePairs2D requires single integer as parameter, not %s'%(type(patch_size)))

    if num_quadrature_pairs is None:
        num_quadrature_pairs = (patch_size**2-1)//2

    R = DFT2(patch_size)
    F = LinearTransform(np.zeros((1+num_quadrature_pairs*2, patch_size**2)), 'DC component + lowest %d quadrature pair Fourier components of %dx%d patches'%(num_quadrature_pairs, patch_size, patch_size))
    F.W[0,:] = R.W[0,:]
    Q = _get_lowest_Fourier_components(patch_size, num_quadrature_pairs)
    F.W[1:,:] = Q.W

    return F

def DCACQuadraturePairs2D_legacy(sh):
    """
    Creates a 2D DFT basis for patch_size X patch_size image patches where the filters with identical spatial
    frequency and orientation are paired up, such that filters 1&2, 3&4, ... are quadrature
    pairs. Filter 0 is the DC component.
    Legacy function. Only for backwards compatibility with old data.

    :param sh: Int that determines the image patch size.
    :type sh: int
    :returns: A linear filter containing rearranged DFT basis, DC filter as first component.
    :rtype: natter.Transforms.LinearTransform
    """

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

    F2 = DFT2(sh)
    F2 = F2[ind,:]
    F2.addToHistory('Rearranged filters into quadrature pairs.')

    return F2

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

def _symmetricOrth( A ):
    """
    Loewdin/symmetric orthogonalization

    :param A: Square matrix
    :type A: numpy.ndarray
    :returns: Orthonormal matrix with smallest Frobenius norm to input matrix
    :rtype: numpy.ndarray
    """

    if A is None:
        return A
    U,d,V = np.linalg.svd(np.dot(A.T, A))
    B = np.dot(A, np.dot(U, np.dot(np.diag(np.sqrt(1/d)),U.T)))
    return B

def _invsqrtm( A ):
    """
    Inverse symmetric matrix square root
    Computed using svd decomposition of A

    :param A: Square matrix
    :type A: numpy.ndarray
    :returns: inverse square root
    :rtype: numpy.ndarray
    """
    U,d,V = np.linalg.svd(A)
    return np.dot(U, np.dot(np.diag(np.sqrt(1/d)),U.T))

def _sqrtm( A ):
    """
    Symmetric matrix square root
    Computed using svd decomposition of A

    :param A: Square matrix
    :type A: numpy.ndarray
    :returns: matrix square root
    :rtype: numpy.ndarray
    """
    U,d,V = np.linalg.svd(A)
    return np.dot(U, np.dot(np.diag(np.sqrt(d)),U.T))

def _SSA_gradient( U, x0, x1 ):
    """
    Gradient of the SSA objective
    will be removed in future releases if SSA becomes part of the
    MDP toolbox

    :param U: Filter matrix
    :type U: numpy.ndarray
    :param x0: Data set at time t=0
    :type x0: numpy.ndarray
    :param x1: Data set at time t=1
    :type x1: numpy.ndarray
    :returns: SSA gradient
    :rtype: numpy.ndarray
    """
    subspace_dim = 2
    kx, n = x0.shape
    k, kx = U.shape
    num_subspaces = k//subspace_dim
    UX0 = np.dot(U,x0)
    UX1 = np.dot(U,x1)

    # now forming \sum{UX0[ii,:]**2} for all subspaces
    SX0 = (UX0.reshape(num_subspaces, subspace_dim, n)**2).sum(1)
    SX1 = (UX1.reshape(num_subspaces, subspace_dim, n)**2).sum(1)

    #creates an index of which component belongs to which subspace
    ind_subspaces = np.repeat(np.arange(num_subspaces),subspace_dim)

    g = np.repeat((.5*((SX0**2).mean(1)+(SX1**2).mean(1)))-subspace_dim**2,2).reshape(k,1)
    f = np.repeat(((SX1-SX0)**2).mean(1),2).reshape(k,1)

    #df = self.gradient_temporalDifferenceSquared(U, x0, x1)
    #dg = self.gradient_meanVariance(U, x0, x1)
    #ind_subspaces = np.repeat(np.arange(num_subspaces),subspace_dim)
    #factor = 4*(SX1[ind_subspaces,:] - SX0[ind_subspaces,:]).reshape(k,1,n)

    #UX0X0 = UX0.reshape(k,1,n)*x0.reshape(1,k,n)
    #UX1X1 = UX1.reshape(k,1,n)*x1.reshape(1,k,n)

    #df = (factor*(UX1X1 - UX0X0)).mean(2)
    #dg = (UX0X0+UX1X1).mean(2)
    dg = np.empty_like(U)
    df = np.empty_like(U)
    factor = 4*(SX1[ind_subspaces,:] - SX0[ind_subspaces,:])
    SUX1 = SX1[ind_subspaces,:]*UX1
    SUX0 = SX0[ind_subspaces,:]*UX0

    for ii in xrange(k):
        dg[ii,:] = (SUX1[ii,:]*x1 + SUX0[ii,:]*x0).mean(1)
        df[ii,:] = (factor[ii,:]*(UX1[ii,:]*x1 - UX0[ii,:]*x0)).mean(1)

    dU = (df*g - f*dg)/g**2

    return dU

def _SSA_objective( U, x0, x1 ):
    """
    objective of the SSA objective
    will be removed in future releases if SSA becomes part of the
    MDP toolbox

    :param U: Filter matrix
    :type U: numpy.ndarray
    :param x0: Data set at time t=0
    :type x0: numpy.ndarray
    :param x1: Data set at time t=1
    :type x1: numpy.ndarray
    :returns: SSA objective
    :rtype: float
    """
    subspace_dim = 2
    kx, n = x0.shape
    k, kx = U.shape
    num_subspaces = k//subspace_dim
    UX0 = np.dot(U,x0)
    UX1 = np.dot(U,x1)
    SX0 = (UX0.reshape(num_subspaces, subspace_dim, n)**2).sum(1)
    SX1 = (UX1.reshape(num_subspaces, subspace_dim, n)**2).sum(1)

    #res = 2*((SX1-SX0)**2).mean(1)/((SX0**2).mean(1)+(SX1**2).mean(1))
    res = 2*((SX1-SX0)).var(1)/((SX0).var(1)+(SX1).var(1))
    return res.mean()

def SSA( dat, verbose=5, maxIterations=1000, minIterations=0, alpha=1, eps=1e-8, *args, **kwargs ):
    """
    Creates a linear filter by applying 2D subspace slowness analysis.

    :param dat: Data set with sequence
    :type dat: natter.DataModule.Data
    :param verbose: Number of iterations between debug outputs (default=5)
    :type verbose: int
    :param maxIterations: Maximum number of iterations
    :type maxIterations: int
    :param minIterations: Minimum number of iterations
    :type minIterations: Minimum number of iterations
    :param alpha: Initial gradient step length
    :type alpha: float
    :param eps: Threshold for stopping the optimization
    :type eps: float
    :param args: Compatibility for old version. Not used anymore.
    :type args: list
    :param kwargs: Compatibility for old version. Not used anymore.
    :type kwargs: dict

    :returns: A linear filter containing the SSA filters
    :rtype: natter.Transforms.LinearTransform
    """
    # mdp code for upcoming SSA integration
    #functionValues = kwargs.pop('functionValues', None)
    #
    #SSA = mdp.nodes.SSANode(input_dim=dat.size(0), **kwargs)
    #SSA.train(dat.X.T)
    #SSA.stop_training()
    #
    #if functionValues:
    #    return LinearTransform(SSA.U,'2D SSA filter computed on ' + dat.name), SSA.functionValues
    #else:
    #    return LinearTransform(SSA.U,'2D SSA filter computed on ' + dat.name)

    x0, x1 = dat.split(2)
    x0 = x0.X
    x1 = x1.X

    if kwargs.has_key('U'):
        U = kwargs['U']
    else:
        U = _symmetricOrth(np.random.randn(dat.dim(), dat.dim()))

    fOld = np.inf
    fNew = _SSA_objective(U, x0, x1)
    counter = 0
    m_init = 0.0
    m_max = 40.0
    m_history = 5
    m_offset = 1
    mset = np.zeros(m_history)
    fVals = np.empty(maxIterations+1)
    fVals[0] = fNew

    while ((fOld-fNew) > eps and counter < maxIterations) \
          or counter < minIterations:
        fOld = fNew
        dU = _SSA_gradient(U, x0, x1)

        m = m_init

        UTemp = _symmetricOrth(U - alpha/(2.**m)*dU)
        fTemp = _SSA_objective(UTemp, x0, x1)

        while fTemp > fOld and m < m_max:
            m += 1
            UTemp = _symmetricOrth(U - alpha/(2.**m)*dU)
            fTemp = _SSA_objective(UTemp, x0, x1)

        mset[np.mod(counter,m_history)] = m
        U = UTemp
        counter += 1
        fNew = fTemp
        fVals[counter] = fNew

        if np.mod(counter,m_history) == 0:
            if mset.mean() == m_init:
                m_init = -5
            else:
                m_init = np.floor(mset.mean())-m_offset

        if not verbose is None and counter%verbose == 0:
            print "Total improvement after iteration " + str(counter) +\
                  ": " + str(fOld-fNew)+" to " + str(fNew) +\
                  ' m: ' + str(m) + '\n'
            print 'Mean m over last %i iterations'%(mset.size) +\
                  ' was %f. Setting new m_init'%(mset.mean()) +\
                  ' to %i\n'%(m_init)

    return LinearTransform(U,'2D SSA filter computed on ' + dat.name)


def mdpWrapper( dat, nodename, output, *args, **kwargs ):
    """
    *EXPERIMENTAL WRAPPER - DO NOT USE IF YOU DON'T KNOW WHAT YOU DO*
    Creates a linear filter by training a mdp node

    :param dat: Data set with sequence
    :type dat: natter.DataModule.Data
    :param nodename: name of the mdp node (without "Node"), case sensitive!
    :type nodename: String
    :param output: Output parameter of the mdp node
    :type output: numpy.ndarray
    :param args: Compatibility for old version. Not used anymore.
    :type args: list
    :param kwargs: Parameters passed directly to mdp node
    :type kwargs: dict
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

    if output[-1] == ')':
        U = getattr(node, output[:output.find('(')])()
    else:
        U = getattr(node, output)


    if functionValues:
        return LinearTransform(U,'2D %s filter computed on '%(nodename) + dat.name), SSA.functionValues
    else:
        return LinearTransform(U,'2D %s filter computed on '%(nodename) + dat.name)

def _get_lowest_Fourier_components( patch_width, component_number ):
    """
    Returns the #component_number lowest quadrature pair Fourier components
    for patch_width-by-patch_width image patches, ordered by spatial
    frequency.

    :param patch_width: edge length of the 2D Fourier components
    :type patch_width: int
    :param component_number: Number of quadrature pairs to return
    :type component_number: int
    :returns: Matrix containing the n quadrature pairs with lowest spatial frequency
    :rtype: numpy.ndarray
    """
    R = DFT2(patch_width)

    if np.mod(patch_width,2) == 1:
        offset = patch_width//2
        X, Y = np.mgrid[-offset:offset+1,-offset:offset+1]
    else:
        offset = patch_width//2
        X, Y = np.mgrid[-offset:offset,-offset:offset]
    origin = np.array((offset,offset))
    center_index = patch_width*offset+offset
    dist = np.sqrt(X**2+Y**2).ravel()[:center_index]
    dist += np.arange(dist.size)*1e-10 #just to make sure that the ordering is constant over all scales, because otherwise 2 equally distant components are randomly ordered
    order = dist.argsort()
    n = component_number
    if component_number > order.size:
        print "There are only %d quadrature pairs available for the given patch size. Returning these."%(order.size)
        n = order.size

    result = LinearTransform(np.zeros((n*2, R.shape[1])))
    result.addToHistory('lowest %d quadrature pair Fourier components of %dx%d patches'%(component_number, patch_width, patch_width))
    for ii in xrange(n):
        py, px = np.array(np.unravel_index(order[ii], (patch_width, patch_width))) - origin
        result.W[2*ii,:] = R.W[np.mod(py,patch_width)*patch_width+np.mod(px,patch_width),:]
        result.W[2*ii+1,:] = R.W[np.mod(-py,patch_width)*patch_width+np.mod(-px,patch_width),:]

    return result

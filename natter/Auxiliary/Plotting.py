import numpy as np
import Errors
from matplotlib import pyplot
import types


def htmltable(rowlab,collab, S):
    """
    Generates a table in html format from the data in S.

    :param rowlab: row labels
    :type rowlab: list
    :param collab: column labels
    :type collab: list
    :param S: list of list of entries for the table. Elements must be convertable to strings via str(.)
    :param S: list
    :returns: html table
    :rtype: string
    """
    b = False
    s = "<table style=\"background-color: rgb(200, 200, 200);\" border=\"1\" cellpadding=\"8\" cellspacing=\"3\" rules=\"cols\" border=\"0\" rules=\"rows\">"
    s += "<tr><td></td>"
    for colelem in collab:
        s += "<td><b>%s</b></td>" % (str(colelem),)

    s += "</tr>"

    for i in xrange(len(rowlab)):
        if b:
            s += "<tr style=\"background-color: rgb(221, 221, 221);\"><td><b>%s</b></td>" % (str(rowlab[i]))
            b = False
        else:
            s += "<tr style=\"background-color: rgb(255, 255, 255);\"><td><b>%s</b></td>" % (str(rowlab[i]))
            b = True
        for j in xrange(len(collab)):
            s += "<td>%s</td>" % (str(S[i][j]),)

        s += "</tr>"

    s+= "</table>"
    return s


def patch2Img(X, nx , ny, orientation='F'):
    """
    Reshapes the patches contained as columns in X and arranges them
    on a nx x ny grid.

    :param X: data matrix
    :type X: numpy.ndarray
    :param nx: row dimension of the grid
    :type nx: int
    :param ny: column dimension of the grid
    :type ny: int
    :param orientation: how the patches are reshaped (default: 'F' which means matlab or fortran column style)
    :type orientation: string
    :returns: 2D images containing the patches as subimages
    :rtype: numpy.ndarray
    """
    sz = np.shape(X)
    p = sz[0]
    n = sz[1]
    ptchSz = np.sqrt(p)
    if int(ptchSz) != ptchSz:
        raise Errors.DimensionalityError('Auxiliary.Plotting.patch2Img: Number of rows is not a square!')
    ptchSz = int(ptchSz)

    imgTmp = np.array(X.reshape( (ptchSz,ptchSz, n),order=orientation ))
    padding_size = tuple([ptchSz,ptchSz,(nx*ny - n)])
    imgPadded = np.dstack((imgTmp, np.zeros( padding_size))).reshape(ptchSz,ptchSz,ny,nx)
    img = np.vstack([np.hstack([imgPadded[:,:,iy,ix] for ix in xrange(nx)]) for iy in xrange(ny)])
    return img

def plotPatches( B , nx, ptchSz, ax=None, contrastenhancement=False, orientation='F', **kwargs):
    """
    Plot columns of A as patches in an array of NX=(dimx,dimy)
    patches. Each patch is assumed to have ptchSz patch size. Further
    kwargs are passed down to imshow.

    :param B: data matrix containing the patches as columns
    :type A: numpy.ndarray
    :param nx: grid dimensions.
    :type nx: tuple of int
    :param ptchSz: patch size
    :type ptchSz: int
    :param ax: axis object in which the patches are plotted if ax is not None.
    :type ax: matplotlib.pyplot.axis
    :param contrastenhancement: boolean that indicates whether the patches should be rescaled for better visibility (default: False)
    :type contrastenhancement: bool
    :param orientation: how the patches are reshaped (default: 'F' which means matlab or fortran column style)
    :type orientation: string
    :param kwargs: Additional parameters directly passed to matplotlib.pyplot.imshow
    :type kwargs: dict
    """
    A = B.copy()

    if type(nx) == types.TupleType:
        if len(nx) == 2:
            ny = int(nx[1])
            nx = int(nx[0])
        else:
            ny = int(nx[0])
            nx = int(nx[0])
    else:
        nx = int(nx)
        ny = int(nx)

    # p = ptchSz**2
    # sz = np.size(A,0)

    A -= np.min(np.min(A))
    A /= np.max(np.max(A))

    # contrast enhancement
    if contrastenhancement:
        A = np.array(A).transpose()
        for k in range(len(A)):
            A[k] -= np.min(A[k])
            A[k] = np.array([((A[k,i] > 10e-10) and A[k,i] or 0.0) for i in range(len(A[k]))])
            A[k] /= np.max(A[k])
        A = A.transpose()


    doShow = False
    if ax == None:
        ax = pyplot
        doShow = True

    I = patch2Img(A, nx, ny, orientation)
    if doShow:
        ax.imshow(I,cmap=pyplot.cm.gray,interpolation='nearest', **kwargs)
    else:
        ax.imshow(I,cmap=pyplot.cm.gray,interpolation='nearest',aspect='auto', **kwargs)
    for i in range(1,nx):
        ax.plot(np.array([i*ptchSz-.5,i*ptchSz-.5]),np.array([-.5,ny*ptchSz-.5]),color='black')
    for i in range(1,ny):
        ax.plot(np.array([-.5,nx*ptchSz-.5]),np.array([i*ptchSz-.5,i*ptchSz-.5]),color='black')

    if doShow:
        ax.axis('tight')
        ax.axis('off')
        #ax.show()


def plotStripes( U, h = None, w = None, clearFigure=True, orientation='C', sameScale=True, plotNumbers=False, **kwargs):
    """
    plot columns of A as 1D line plots in an array of NX=(h,w) subsplots. Takes all arguments that pyplot.plot takes.

    :param U: 2D Filter matrix
    :type U: numpy.ndarray
    :param h: number of subplot rows. Will be estimated if not given.
    :type h: integer
    :param w: number of subplot columns. Will be estimated if not given.
    :type w: integer
    :param clearFigure: Indicates if the figure should be cleared before plotting (set to False if you want to plot several filter matrices in one figure). Default: True
    :type clearFigure: bool
    :param orientation: Signals whether row-major ('C') or column-major ('F') plotting should be used. Default 'C'.
    :type orientation: string
    :param sameScale: Indicates if all subplots should have the same x- and y-axis limits. Default True.
    :type sameScale: bool
    :param plotNumbers: Plot the subplot number as title of the subplot. Default False.
    :type plotNumbers: bool
    :param kwargs: All arguments which pyplot.plot() accepts
    :type kwargs: dict
    """

    if h == None or w == None:
        try:
            [h, w] = findShape(U.shape[1])
        except:
            [h, w] = findShape(U.shape[1]+1)

    if clearFigure:
        pyplot.clf()

    ind = (np.arange(U.shape[1])+1).reshape(h,w).flatten(orientation)
    if len(pyplot.gcf().axes) == 0 or clearFigure:
        ymax = U.max()
        ymin = U.min()
    else:
        ymax = max(U.max(), pyplot.ylim()[1])
        ymin = min(U.min(), pyplot.ylim()[0])

    for ii in xrange(int(h)):
        for jj in xrange(int(w)):
            pyplot.subplot(h,w,ind[ii*w+jj])
            if plotNumbers:
                pyplot.title(str(ind[ii*w+jj]))
            if ii*w+jj < U.shape[1]:
                pyplot.plot(U[:,ii*w+jj], **kwargs)
                if sameScale:
                    pyplot.ylim((ymin, ymax))


def findShape( s ):
    """
    Tries to find two factors h,w which fulfill h*w=s and min|h-w| such that
    a vector with length s can be reshaped into a hXw 2D matrix. If h != w then
    w>h, so the matrix will be in landscape format. Throws a ValueError if s
    is prime.

    :param s: integer to be factorized
    :type s: int
    :returns: A tuple (h,w) which factorizes s
    :rtype: tuple of int
    """
    w = float(np.ceil(np.sqrt(s)))
    h = s / w
    while h != round(h) and h > 2:
        w += 1
        h = s / w
    if h != round(h):
        raise ValueError, "Image vector is not reshapeable into a 2D image. (pixel number is prime)"
    return int(h),int(w)


def savefig( filename, bb='tight' ):
    """
    Saves the current figure to filename in png and eps format with 300 dpi.

    :param filename: filename of the file the figure is saved to.
    :type filename: string
    :param bb: bbox_inches parameter of pyplot.savefig.
    :type bb: string
    """
    print 'Saving figure %s'%(filename)
    pyplot.savefig('%s.png'%(filename), bbox_inches=bb, dpi=300)
    pyplot.savefig('%s.eps'%(filename), bbox_inches=bb, dpi=300)

def imsave( filename, image ):
    """
    Saves the given 2D array to filename in png and eps format

    :param filename: filename of the file the array is saved to.
    :type filename: string
    :param image: 2D array
    :type image: numpy.ndarray
    """
    print 'Saving image %s'%(filename)
    pyplot.imsave('%s.png'%(filename), image)
    pyplot.imsave('%s.eps'%(filename), image)

import numpy as np
import Errors
from matplotlib import pyplot
import types


def htmltable(rowlab,collab, S):
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


def patch2Img(X, nx , ny):
    sz = np.shape( X)
    p = sz[0]
    n = sz[1]
    ptchSz = np.sqrt( p)
    if int(ptchSz) != ptchSz:
        raise Errors.DimensionalityError('Auxiliary.Plotting.patch2Img: Number of rows is not a square!') 

    imgTmp = np.array(X.reshape( (ptchSz,n*ptchSz),order="F" ))
    newSz = (ny*ptchSz, nx*ptchSz)
    img = np.zeros( newSz);
    newSz = tuple([ptchSz,ptchSz*( nx*ny - n)])
    imgTmp = np.concatenate((imgTmp, np.zeros( newSz)),1);
    for ii in range(ny):
        img[ ii*ptchSz:(ii+1)*ptchSz, :] = imgTmp[ :, ii*nx*ptchSz:(ii+1)*nx*ptchSz]
    return img
 


def plotPatches( B , nx, ptchSz,ax=None,contrastenhancement=False):
    """
    PLOTPATCHES(A, NX, PTCHSZ)

    plot columns of A as patches in an array of NX=(dimx,dimy) patches. Each patch is assumed to have PTCHSZ patch size. 
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
        
    I = patch2Img(A, nx,ny)
    if doShow:
        ax.imshow(I,cmap=pyplot.cm.gray,interpolation='nearest')
    else:
        ax.imshow(I,cmap=pyplot.cm.gray,interpolation='nearest',aspect='auto')
    for i in range(1,nx):
        ax.plot(np.array([i*ptchSz-.5,i*ptchSz-.5]),np.array([-.5,ny*ptchSz-.5]),color='black')
    for i in range(1,ny):
        ax.plot(np.array([-.5,nx*ptchSz-.5]),np.array([i*ptchSz-.5,i*ptchSz-.5]),color='black')

    if doShow:
        ax.axis('tight')
        ax.axis('off')
        #ax.show()
    
    

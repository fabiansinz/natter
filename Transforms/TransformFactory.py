from __future__ import division
import numpy as np
import mdp
from LinearTransform import LinearTransform
from NonlinearTransform import NonlinearTransform
from numpy import linalg
from scipy import special
import Distribution
from Auxiliary import Errors
import Data
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
            
    


################################ NONLINEAR FILTERS ########################################

def RadialFactorization(psource):
    if not isinstance(psource,Distribution.LpSphericallySymmetric):
        raise TypeError('Transform.TransformFactory.RadialFactorization: psource must be a Lp-spherically symmetric distribution')
    ptarget = Distribution.LpGeneralizedNormal({'s':(special.gamma(1.0/psource.param['p'])/special.gamma(3.0/psource.param['p']))**(psource.param['p']/2.0),\
                                                'p':psource.param['p']})
    print ptarget
    return RadialTransformation(psource,ptarget)

def RadialTransformation(psource,ptarget):
    if ptarget.param['p'] != psource.param['p']:
        raise Errors.ValueError('Values of p must agree')
    p = ptarget.param['p']
    name = 'Radial Factorization Transform ' + psource.name + ' (' + psource.param['rp'].name + ') --> ' + ptarget.name + ' (' + ptarget.param['rp'].name + ')'
    psource = psource.param['rp'].copy()
    ptarget = ptarget.param['rp'].copy()
    g = lambda x: x.scaleCopy( ptarget.ppf(psource.cdf(x.norm(p))).X / x.norm(p).X)
    gdet = lambda y: logDetJacobianRadialTransform(y,psource,ptarget,p)
    return NonlinearTransform(g,name,logdetJ=gdet)

def logDetJacobianRadialTransform(dat,psource,ptarget,p):
    r = dat.norm(p)
    r2 = ptarget.ppf(psource.cdf(r))
    n = dat.size(0)
    return (n-1)*np.log(r2.X) - (n-1)*np.log(r.X) + psource.loglik(r) - ptarget.loglik(r2)
    
#################################################

def LpNestedNonLinearICA(p):
    L = p.param['f'].copy()
    rp = p.param['rp'].copy()

    return getLpNestedNonLinearICARec(rp,L,())
    
def getLpNestedNonLinearICARec(rp,L,mind):
    p = L.p[L.pdict[()]]
    s = (special.gamma(1.0/p) / special.gamma(3.0/p))**(p/2.0)
#    s = rp.param['s']
    rptarget = Distribution.GammaP({'u':float(L.n[()])/L.p[L.pdict[()]],'s':s, 'p':L.p[L.pdict[()]]})
    g = lambda x: x.scaleCopy( rptarget.ppf(rp.cdf(L(x))).X / L(x).X, L.iByI[()])
    gdet = lambda y: logDetJacobianLpNestedTransform(y,rp,rptarget,L)
    F = NonlinearTransform(g,'Rescaling of ' + str(mind),logdetJ=gdet)

    for k in range(L.l[()]):
        if L.n[(k,)] > 1:
            L2 = L[(k,)]
            pnew = Distribution.GammaP({'u':float(L2.n[()])/L.p[L.pdict[()]],'s':s, 'p':L.p[L.pdict[()]]})
            F2 = getLpNestedNonLinearICARec(pnew,L2,mind + (k,))
            F = F2*F


    return F
    
def logDetJacobianLpNestedTransform(dat,psource,ptarget,L):
    r = L(dat)
    r2 = ptarget.ppf(psource.cdf(r))
    n = L.n[()]
    return (n-1)*np.log(r2.X) - (n-1)*np.log(r.X) + psource.loglik(r) - ptarget.loglik(r2)

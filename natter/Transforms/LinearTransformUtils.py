from __future__ import division
from numpy.fft import fft2, fftshift
from numpy import reshape, sqrt, argmax, arange,ceil, max, abs, arctan2, meshgrid, array, imag, real, pi, mean, vstack, cos, dot, zeros, sin,where
# from scipy.optimize import fmin_bfgs, fmin_cg
# from numpy.random import randn
from natter.Auxiliary.Statistics import quantile


def bestMatchingGratings(F):
    W = F.W
    N = sqrt(W.shape[1])
    f = arange(-ceil(N/2),ceil(N/2))
    Nx,Ny = meshgrid(arange(N),arange(N))
    # Nxy = vstack( (Nx.flatten('F'), Ny.flatten('F')) )
    
    ret = []
    for k in xrange(W.shape[0]):
        w = reshape(W[k,:],(N,N),order='F')

        # extract center
        pw = abs(w)
        qu = quantile(pw.flatten(),0.5)
        pw[where(pw <= qu)] = 0.0
        pw = pw/sum(pw.flatten())
        mu = array([sum(sum(pw*Nx)), sum(sum(pw*Ny))])

        # extract maximal frequency
        z = fftshift(fft2(w))
        az = abs(z)
        i = argmax(max(az,0),0)
        j = argmax(az[:,i])
        omega = array([f[i],f[j]])

        # extract phase
        phi = arctan2(imag(z[i,j]),real(z[i,j])) + 2*pi/N*(f[i]*mu[0]+f[j]*mu[1])

        # refine
        # x0 = zeros((3,))
        # x0[:2] = omega
        # x0[2] = phi
        # print x0
        # x0 = fmin_bfgs(_optfunc,x0,fprime=_doptfunc,args=(w,Nxy,mu,N),gtol=1e-15)
        # print x0
        # raw_input()

        ret.append({'frequency':omega,\
                    'phase': phi, \
                    'center': mu})
        
    return ret


def _optfunc(x,w,Nxy,mu,N):
    return mean( (w.flatten('F')-cos(2*pi/N *dot(x[:2],Nxy) - 2*pi/N*sum(mu*x[:2]) + x[2]))**2.0 )


def _doptfunc(x,w,Nxy,mu,N):
    ret = zeros((3,))
    tmp = w.flatten('F')
    tmp2 = Nxy - reshape(mu,(2,1))
    ret[:2] = -2*mean( (tmp-cos(2*pi/N *dot(x[:2],tmp2) + x[2]))* \
                       sin(2*pi/N *dot(x[:2],tmp2) + x[2])*2*pi/N*tmp2 ,1)
    
    ret[2] = -2*mean( (tmp-cos(2*pi/N *dot(x[:2],tmp2)  + x[2]))* \
                       sin(2*pi/N *dot(x[:2],tmp2)  + x[2]))
    return ret

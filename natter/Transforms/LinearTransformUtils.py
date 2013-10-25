from __future__ import division
from numpy.fft import fft2, fftshift,fft
from numpy import reshape, sqrt, argmax, arange,ceil, max, abs, arctan2, meshgrid, array, imag, real, pi, mean, vstack, cos, dot, sin,where,percentile, zeros
from scipy.optimize import fmin_bfgs
# from numpy.random import randn


def getTuningCurve(R,fundamental_freq=1):
    """
    Extracts the tuning curve from the responses R by taking the
    amplitude of the response at the fundamental frequency.

    :param R: Array containing the responses to a grating moving with the fundamental frequency in each row.
    :type R: numpy.array
    :param fundamental_freq: Fundamental frequency number
    :type fundamental_freq: int
    :returns: the values for the tuning curve
    :rtype: numpy.array

    """

    ret = zeros((R.shape[0],))
    N = R.shape[1]
    for k in xrange(R.shape[0]):
        ret[k] = 2/N*abs(fft(R[k,:])[fundamental_freq])
    return ret


def bestMatchingGratings(F):
    """
    Determines the parameters of the best matching gratings of the filers in F.

    :param F: filters
    :type F: natter.Transform.LinearTransform
    :returns: list of dictionaries containing the parameters
    :rtype: list
    """
    W = F.W
    N = sqrt(W.shape[1])
    f = arange(-ceil(N/2),ceil(N/2))
    Nx,Ny = meshgrid(arange(N),arange(N))
    Nxy = vstack( (Nx.flatten('F'), Ny.flatten('F')) )

    ret = []
    for k in xrange(W.shape[0]):
        w = reshape(W[k,:],(N,N),order='F')

        # extract center
        pw = abs(w)
        qu = percentile(pw.flatten(),80)
        pw[where(pw <= qu)] = 0.0
        pw = pw/sum(pw.flatten())
        mu = array([sum(sum(pw*Nx)), sum(sum(pw*Ny))])

        # extract maximal frequency
        z = fftshift(fft2(w))
        az = abs(z)

        j = argmax(max(az,0),0)
        i = argmax(az[:,j])
        omega = array([f[j],f[i]])

        # extract phase
        phi = arctan2(imag(z[i,j]),real(z[i,j])) + 2*pi/N*dot(mu,omega)

        # refine
        x0 = zeros((3,))
        x0[:2] = omega
        x0[2] = phi
        #print x0
        x0 = fmin_bfgs(_optfunc,x0,fprime=_doptfunc,args=(w,Nxy,mu,N),gtol=1e-20)
        #print x0
        # raw_input()
        omega = x0[:2]
        phi = x0[2]

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

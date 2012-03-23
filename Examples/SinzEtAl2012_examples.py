#!/usr/bin/env python
from __future__ import division

from matplotlib.pyplot import show, figure

from natter.DataModule import DataSampler, DataLoader
from natter.Auxiliary import ImageUtils
from natter.Transforms import LinearTransformFactory, NonlinearTransformFactory
from natter.Distributions import ProductOfExponentialPowerDistributions,\
     Distribution, LpSphericallySymmetric, CompleteLinearModel
from natter.Auxiliary.Utils import parseParameters
from natter.Logging import Table, ExperimentLog


# Loading a simple Data module from an ascii file
dat = DataLoader.load('hateren4x4_train_No1.dat')
print dat



# Sampling 50000 4x4 patches from all images in the DATADIR using the DataSampler
DATADIR = '/share/data/hateren/iml/'
loadFunc = ImageUtils.loadHaterenImage
sampleFunc =  DataSampler.img2PatchRand
noSamples = 50000
patchSize = 4
myIter = DataSampler.directoryIterator(DATADIR, noSamples, patchSize, loadFunc, sampleFunc)
dat = DataSampler.sample(myIter,noSamples)

# Setting the mean of the data set over samples and dimensions to 0 and storing
# the mean value in mu (1D float)
mu = dat.center()
print "data object is now centered:\n", dat


# In most cases, the mean lightness of a patch (DC component) is not of interest
# and will therefor be removed/projected out. We first create a orthonormal
# basis of which the first filter has identical entries everywhere (DC filter)
# and all other filters are orthonormal (AC filters) then we multiply the data
# to the n-1 remaining AC filters
FDCAC = LinearTransformFactory.DCAC(dat)
print "DC/AC filter:\n", FDCAC

# first way: first applying all filters to data then removing DC component (inefficient)
dat = FDCAC*dat
dat = dat[1:,:]
# second way: removing DC component from filter and then applying it to data (efficient)
FAC = FDCAC[1:,:]
dat = FAC*dat
print "Data with DC component projected out:\n", data

# The data usually has different variances in different dimensions as well as
# correlations between different dimensions. By applying a whitening operation
# to the data, the correlations are removed and the data covariance matrix
# will be the identity matrix.
FwPCA = LinearTransformFactory.wPCA(dat)
dat = FwPCA*dat

D = dat.makeWhiteningVolumeConserving()


pICA = ProductOfExponentialPowerDistributions(n=dat.dim())
print pICA

N = pICA['n']
q = pICA['P'][0]

pICA['P'][0]

pICA.estimate(dat)
pICA['P'][0]


pICA['P'][0].histogram(dat[0,:],bins=100)
show()

fig = figure()
ax = fig.add_subplot(111)
dat[:2,:].plot(ax = ax)
dat2 = pICA.sample(50000)
dat[:2,:].plot(ax = ax,color='b')
dat2[:2,:].plot(ax=ax,color='r')
ax.set_xlabel(r'$y_1$',fontsize=14)
ax.set_ylabel(r'$y_2$',fontsize=14)
ax.set_title(r'Scatter plot')
ax.axis([-15.,15.,-15.,15.])
show()

pLp.estimate(dat)
pLp['rp'].histogram(dat.norm(p['p']),bins=100)
fig = figure()
ax =fig.add_subplot(111)
dat[:2,:].plot(ax = ax)
dat2 = pLp.sample(50000)
dat[:2,:].plot(ax = ax,color='b')
dat2[:2,:].plot(ax=ax,color='r')
ax.set_xlabel(r'$y_1$',fontsize=14)
ax.set_ylabel(r'$y_2$',fontsize=14)
ax.set_title(r'Scatter plot')
ax.axis([-15.,15.,-15.,15.])
show()



class myDistribution(Distribution):

    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)

        # set default parameters
        self.name = 'My Distribution'
        self.param = {'a':1.0,'b':1.0} # default parameters
        if param is not None:
            for k in param.keys():
                self.param[k] = float(param[k])
        self.primary = ['a','b']

pLp = LpSphericallySymmetric(n=dat.dim())

pLp.estimate(dat)
pLp['rp'].histogram(dat.norm(p['p']),bins=100)
fig = figure()
ax =fig.add_subplott(111)
dat[:2,:].plot(ax = ax)
dat2 = pLp.sample(50000)
dat[:2,:].plot(ax = ax,color='b')
dat2[:2,:].plot(ax=ax,color='r')
ax.set_xlabel(r'$y_1$',fontsize=14)
ax.set_ylabel(r'$y_2$',fontsize=14)
ax.set_title(r'Scatter plot')
ax.axis([-15.,15.,-15.,15.])
show()

n = dat.dim()
pCLM = CompleteLinearModel(n=n,q=LpSphericallySymmetric(n=n))
print pCLM

W = LinearTransformFactory.fastICA(dat)
n = dat.dim()
pCLM = CompleteLinearModel(n=n,q=LpSphericallySymmetric(n=n),W=W)
print pCLM
pCLM.estimate(dat[:,:3000])
fig = figure()
ax =fig.add_subplott(111)
dat[:2,:].plot(ax = ax)
dat2 = pCLM.sample(50000)
dat[:2,:].plot(ax = ax,color='b')
dat2[:2,:].plot(ax=ax,color='r')
ax.set_xlabel(r'$y_1$',fontsize=14)
ax.set_ylabel(r'$y_2$',fontsize=14)
ax.set_title(r'Scatter plot')
ax.axis([-15.,15.,-15.,15.])
fig = figure()
ax =fig.add_subplott(111)
F = FDC.stack(pCLM['W']*FwPCA*FAC)
F.plotFilters(ax=ax)
show()

pLp = LpSphericallySymmetric(n=n,p=1.3)
pLp.primary.remove('p') \# exclude p from the estimation
pCLM = CompleteLinearModel(n=n,q=pLp,W=W)

pLp = LpSphericallySymmetric(n=dat.dim())
pICA = ProductOfExponentialPowerDistributions(n=dat.dim())
pLp.estimate(dat)
pICA.estimate(dat)
FRF = NonlinearTransformFactory.RadialFactorization(pLp)
dat2 = FRF*dat
pICA2 = pICA.copy()
pICA2.estimate(dat2)
fig = figure()
ax =fig.add_subplott(111)
dat3 = pICA.sample(50000)
dat[:2,:].plot(ax=ax, color='b')
dat3[:2,:].plot(ax=ax, color='r')
fig = figure()
ax =fig.add_subplott(111)
dat2[:2,:].plot(ax=ax)
dat3 = pICA2.sample(50000)
dat2[:2,:].plot(ax=ax, color='b')
dat3[:2,:].plot(ax=ax, color='r')
show()

datTest = DataLoader.load('hateren4x4_test_No1.dat')
datTest.center(mu)
datTest.makeWhiteningVolumeConserving(D=s)
datTest = FwPCA*FAC*datTest
pLp = LpSphericallySymmetric(n=dat.dim())
pICA = ProductOfExponentialPowerDistributions(n=dat.dim())
pLp.estimate(dat)
pICA.estimate(dat)
T = Table(['train ALL','test ALL'],['factorial','Lp-spherically symmetric'])
T['train ALL','factorial'] = pICA.all(dat)
T['test ALL','factorial'] = pICA.all(datTest)
T['train ALL','Lp-spherically symmetric'] = pLp.all(dat)
T['test ALL','Lp-spherically symmetric'] = pLp.all(datTest)
print T

from natter.Logging import
p = ExperimentLog('My fancy experiment')
p += 'We sampled of data we found on the website:'
p *= ('http://dataparadise.com','data paradise')
p.write('results.html',format='html')

U = LinearTransformFactory.mdpWrapper(dat, 'FastICA', output='filters', verbose=True, whitened=True)
dat_ica = U*dat
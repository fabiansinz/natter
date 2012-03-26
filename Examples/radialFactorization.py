from natter.DataModule import DataLoader
from natter.Transforms import LinearTransformFactory
from natter.Distributions import LpSphericallySymmetric,ProductOfExponentialPowerDistributions
from natter.Transforms import NonlinearTransformFactory
from matplotlib.pyplot import show, figure, title, legend

# Data loading and preprocessing as in the previous example
dat = DataLoader.load('hateren4x4_train_No1.dat.gz')
mu = dat.center()
FDCAC = LinearTransformFactory.DCAC(dat)
s = dat.makeWhiteningVolumeConserving()
FDC = FDCAC[0,:]
FAC = FDCAC[1:,:]
FwPCA = LinearTransformFactory.wPCA(FAC*dat)
dat = FwPCA*FAC*dat
print "Data after preprocessing:\n", dat

# Now we create an Lp-spherical symmetric distribution with default parameters
# and a product of exponential power distributions (an approach that ICA [Independent
# component analysys] uses)
pLp = LpSphericallySymmetric(n=dat.dim())
pICA = ProductOfExponentialPowerDistributions(n=dat.dim())
print "Lp-spherical symmetric distribution before fitting:\n", pLp
print "Product of exponential power distributions before fitting:\n", pICA

# Then we estimate the parameters of the distributions based on the given data
pLp.estimate(dat)
pICA.estimate(dat)
print "Lp-spherical symmetric distribution after fitting:\n", pLp
print "Product of exponential power distributions after fitting:\n", pICA

# Next, we compute a nonlinear transform called "radial factorization" which
# maps the radial components of the data to Lp-generalized normal distributions
# For details see documentation.
FRF = NonlinearTransformFactory.RadialFactorization(pLp)
print "Radial factorization:\n", FRF

# Now we create a second data set by applying the radial factorization to the
# source data and estimate a second product of exponential power distributions
# on the nonlinearly transformed data
dat2 = FRF*dat
pICA2 = pICA.copy()
pICA2.estimate(dat2)

# Finally, we sample 50000 data points from the ICA-like distributions both
# with (pICA2) and without (pICA) radial factorization and plot them against
# the true data.
fig = figure(1)
ax = fig.add_axes([.1,.1,.8,.8])
dat3 = pICA.sample(50000)

dat[:2,:].plot(ax = ax,color='b',label='true data')
dat3[:2,:].plot(ax=ax,color='r',label='sampled data')
title('Data and samples without radial factorization')
legend()

fig = figure(2)
ax = fig.add_axes([.1,.1,.8,.8])
dat2[:2,:].plot(ax = ax)
dat4 = pICA2.sample(50000)

dat2[:2,:].plot(ax = ax,color='b',label='true data')
dat4[:2,:].plot(ax=ax,color='r',label='sampled data')
title('Data and samples with radial factorization')
legend()
show()

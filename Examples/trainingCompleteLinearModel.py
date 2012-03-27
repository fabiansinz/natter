from natter.DataModule import DataLoader
from natter.Transforms import LinearTransformFactory
from natter.Distributions import LpSphericallySymmetric, CompleteLinearModel
from matplotlib.pyplot import show, figure, legend

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

# Now we learn a complete linear model which consists of a distribution q and
# a linear transform W. We first learn the linear Transform using fastICA
# and then create the complete linear model with a Lp-spherical symmetric distribution
W = LinearTransformFactory.fastICA(dat)
n = dat.dim()
pCLM = CompleteLinearModel(n=n,q=LpSphericallySymmetric(n=n),W=W)
print "Complete linear model before parameter fitting:\n", pCLM

# Next the parameters of the CLM are estimated from the first 3000 data points
# (otherwise it might take a long time)
pCLM.estimate(dat[:,:3000])
print "Complete linear model after parameter fitting:\n", pCLM


# Now, we sample 50000 data points from the fitted CLM and plot
# them in a scatter plot against the true data (only first 2 dimensions)
# The data is quite well matched.
fig = figure(1)
ax = fig.add_axes([.1,.1,.8,.8])
dat2 = pCLM.sample(50000)
dat[:2,:].plot(ax = ax,color='b',label='true data')
dat2[:2,:].plot(ax=ax,color='r',label='sampled data')
ax.set_xlabel(r'$y_1$',fontsize=14)
ax.set_ylabel(r'$y_2$',fontsize=14)
ax.set_title(r'Scatter plot')
ax.axis([-15.,15.,-15.,15.])
legend()

# Finally, we plot the linear filters of the CLM. For that we have to combine
# all filters we applied to the data (DC-removal, whitening, and CLM) by
# multiplying them together. Otherwise the filters would not be in pixel space.
# The missing/blank filter is the DC component
fig = figure(2)
ax = fig.add_axes([.1,.1,.8,.8])
F = FDC.stack(pCLM['W']*FwPCA*FAC)
F.plotFilters(ax=ax)
show()

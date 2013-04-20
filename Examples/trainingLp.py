from natter.DataModule import DataLoader
from natter.Transforms import LinearTransformFactory
from natter.Distributions import LpSphericallySymmetric
from matplotlib.pyplot import show, figure, legend

# Data loading and preprocessing as in the previous examples
dat = DataLoader.load('hateren8x8_train_No1.dat.gz')
mu = dat.center()
FDCAC = LinearTransformFactory.DCAC(dat)
s = dat.makeWhiteningVolumeConserving()
FDC = FDCAC[0,:]
FAC = FDCAC[1:,:]
FwPCA = LinearTransformFactory.wPCA(FAC*dat)
dat = FwPCA*FAC*dat
print "Data after preprocessing:\n", dat

# Now we create an Lp-spherical distribution with default parameters
p = LpSphericallySymmetric(n=dat.dim())
print "Lp spherically symmetric distribution before parameter estimation", p

# now we estimate the primary parameters of the distribution (the exponent p and
# the recursively the primary parameters of the radial distribution)
p.estimate(dat)
print "Lp spherically symmetric distribution after parameter estimation", p

# This prints a histogram of the data against the fitted radial distribution
figure(1)
p['rp'].histogram(dat.norm(p['p']),bins=100)

# Finally, we sample 50000 data points from the fitted distribution and plot
# them in a scatter plot against the true data (only first 2 dimensions)
# The data is quite well matched.
fig = figure(2)
ax = fig.add_axes([.1,.1,.8,.8])
dat2 = p.sample(50000)
dat[:2,:].plot(ax = ax,color='b',label='true data')
dat2[:2,:].plot(ax=ax,color='r',label='sampled data')
ax.set_xlabel(r'$y_1$',fontsize=14)
ax.set_ylabel(r'$y_2$',fontsize=14)
ax.set_title(r'Scatter plot')
ax.axis([-15.,15.,-15.,15.])
legend()
show()

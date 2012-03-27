from natter.DataModule import DataLoader
from natter.Transforms import LinearTransformFactory
from natter.Distributions import ProductOfExponentialPowerDistributions
from matplotlib.pyplot import show, figure, legend

# Data loading and preprocessing as in the previous examples
dat = DataLoader.load('hateren4x4_train_No1.dat.gz')
mu = dat.center()
FDCAC = LinearTransformFactory.DCAC(dat)
s = dat.makeWhiteningVolumeConserving()
FDC = FDCAC[0,:]
FAC = FDCAC[1:,:]
FwPCA = LinearTransformFactory.wPCA(FAC*dat)
dat = FwPCA*FAC*dat
print "Data after preprocessing:\n", dat

# Now we create a product of exponential power distributions (an approach that
# ICA [Independent component analysys] uses)
p = ProductOfExponentialPowerDistributions(n=dat.dim())
print "Product of exponential power distributions before fitting:\n", pICA

# The primary parameters are the number of distributions n (we take 1 distribution
# per input dimension) and the set of exponential distributions P.
# We estimate the parameters of each exponential distribution based on the given data
p.estimate(dat)
print "Product of exponential power distributions after fitting:\n", pICA

# To inspect the quality of the fit we plot the histogram of the data against
# the pdf of the exponential distribution. Here, we look only at the first dimension
p['P'][0].histogram(dat[0,:],bins=100)

# Finally, we sample 50000 data points from the fitted distributions and plot
# them in a scatter plot against the true data (only first 2 dimensions)
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

from natter.DataModule import DataLoader
from natter.Transforms import LinearTransformFactory

dat = DataLoader.load('hateren8x8_train_No1.dat.gz')
# Setting the mean of the data set over samples and dimensions to 0 and storing
# the mean value in mu (1D float)
mu = dat.center()

# In most cases, the mean lightness of a patch (DC component) is not of interest
# and will therefor be removed/projected out. We first create a orthonormal
# basis of which the first filter has identical entries everywhere (DC filter)
# and all other filters are orthonormal (AC filters) then we multiply the data
# to the n-1 remaining AC filters
FDCAC = LinearTransformFactory.DCAC(dat)
print "DC/AC filter:\n", FDCAC

# Rescales the data such that the whitening matrix (here the matrix square root
# of the data covariance matrix) has determinant one, i.e. the volume of the
# data does not change.
s = dat.makeWhiteningVolumeConserving()
print dat

# removing DC component from filter and then applying it to the data object
FDC = FDCAC[0,:]
FAC = FDCAC[1:,:]
dat = FAC*dat
print "Data with DC component projected out:\n", dat

# Here we plot the first two dimensions of the data object as a 2D scatter
# plot. The two components are highly correlated and have different variances.
from matplotlib.pyplot import show, figure
figure(1)
dat[0:2,:].plot()
show()

# The data usually has different variances in different dimensions as well as
# correlations between different dimensions. By applying a whitening operation
# to the data, the correlations are removed and the data covariance matrix
# will be the identity matrix.
FwPCA = LinearTransformFactory.wPCA(dat)
dat = FwPCA*dat
print "Whitened data:\n", dat

print "The determinant is: "
print (FDC.stack(FwPCA*FAC)).det()

# Here we plot the first two dimensions of the data object as a 2D scatter
# plot to visualize how the distribution has changed by whitening so that
# the two components are now uncorrelated and have variance 1.
from matplotlib.pyplot import show, figure
figure(2)
dat[0:2,:].plot()
show()

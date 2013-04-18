from natter.DataModule import DataLoader
from natter.Transforms import LinearTransformFactory
from natter.Distributions import LpSphericallySymmetric,ProductOfExponentialPowerDistributions
from natter.Logging import Table

# Data loading and preprocessing as in the previous examples
dat = DataLoader.load('hateren8x8_train_No1.dat.gz')
mu = dat.center()
FDCAC = LinearTransformFactory.DCAC(dat)
s = dat.makeWhiteningVolumeConserving()
FDC = FDCAC[0,:]
FAC = FDCAC[1:,:]
FwPCA = LinearTransformFactory.wPCA(FAC*dat)
dat = FwPCA*FAC*dat
print "Training data after preprocessing:\n", dat

# Now we load a second data set, the test set, and center it using the mean value
# we obtained from the training set before. The volume conserving transformation
# is also applied using the same transformation as before
datTest = DataLoader.load('hateren8x8_test_No1.dat.gz')
datTest.center(mu)
datTest.makeWhiteningVolumeConserving(D=s)
datTest = FwPCA*FAC*datTest
print "Test data after preprocessing:\n", datTest

# Now we create an Lp-spherical symmetric distribution with default parameters
# and a product of exponential power distributions (an approach that ICA [Independent
# component analysys] uses)
pLp = LpSphericallySymmetric(n=dat.dim())
pICA = ProductOfExponentialPowerDistributions(n=dat.dim())
print "Lp-spherical symmetric distribution before fitting:\n", pLp
print "Product of exponential power distributions before fitting:\n", pICA

# Then we estimate the parameters of the distributions based on the given training data
pLp.estimate(dat)
pICA.estimate(dat)
print "Lp-spherical symmetric distribution after fitting:\n", pLp
print "Product of exponential power distributions after fitting:\n", pICA

# Noe we create a table to compare the average log-loss (all) of the training
# and test data on the two trained models.
T = Table(['train ALL','test ALL'],['factorial','Lp-spherically symmetric'])
T['train ALL','factorial'] = pICA.all(dat)
T['test ALL','factorial'] = pICA.all(datTest)
T['train ALL','Lp-spherically symmetric'] = pLp.all(dat)
T['test ALL','Lp-spherically symmetric'] = pLp.all(datTest)

print "Table with test results:\n",T




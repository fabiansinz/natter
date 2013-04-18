# Loading from files
from natter.DataModule import DataLoader
print "Loading a simple Data module from an ascii file"
FILENAME = 'hateren8x8_train_No1.dat.gz'
dat = DataLoader.load(FILENAME)
print dat

#------------------------------
# Sampling 50000 4x4 patches from all images in the DATADIR using the DataSampler
from natter.DataModule import DataSampler
from natter.Auxiliary import ImageUtils
print "Sample data from images"

# set DATADIR to the location where you have stored the van Hateren images
# the image database can be downloaded at http://bethgelab.org/datasets/vanhateren/
DATADIR = '/data/hateren/'
# loadHaterenImage provides a loading routine for the van Hateren image format
# but any method that takes a filename and returns a 2D numpy array can be used
# e.g. matplotlib.pyplot.imread
loadFunc = ImageUtils.loadHaterenImage
# The sample function defines how the patches are sampled from the larger image
# img2PatchRand just takes patches from random positions from all over the image
sampleFunc =  DataSampler.img2PatchRand
numSamples = 50000
patchSize = 8
# The directory iterator samples the same amount of patches from every image in
# DATADIR. Every time the iterator is called it returns a new patch. For more
# information on iterators see python documentation. The iterator can easily be
# altered to e.g. not sample from certain image regions or discard patches with
# unwanted features without having to alter the sampling.
myIter = DataSampler.directoryIterator(DATADIR, numSamples, patchSize, loadFunc, sampleFunc)
# The sample method now calls the iterator numSamples times and creates the data
# object with all the sampled patches
dat = DataSampler.sample(myIter, numSamples)

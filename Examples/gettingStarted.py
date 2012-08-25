from __future__ import division
from sys import stdin
import numpy as np

from natter.DataModule import Data
from natter.Transforms import LinearTransformFactory

patch_dim = 8
num_samples = 10000

dat = Data(X=np.random.randn(patch_dim**2, num_samples), name='Gaussian white noise')

U = LinearTransformFactory.SSA(dat, maxIterations=50)

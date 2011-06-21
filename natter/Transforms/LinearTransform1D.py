import LinearTransform
from numpy.linalg import inv
from natter.Auxiliary import Plotting

class LinearTransform1D(LinearTransform.LinearTransform):
    """
    LinearTransform1D class

    Implements linear transforms of data with 1D filters.
    
    :param W: Matrix for initializing the linear transform.
    :type W: numpy.array 
    :param name: Name of the linear transform
    :type name: string
    :param history: List of previous operations on the linear transform.
    :type history: List of (lists of) strings.
    
        
    """
    def plotBasis(self, plotNumbers=False, **kwargs):
        """

        Plots the columns of the inverse linear transform matrix
        W as 1D plot. 

        :param plotNumbers: Determines whether the index of the basis function should be plotted as well.
        :type plotNumbers: bool
        :param **kwargs: See natter.Auxiliary.Plotting.plotStripes

        """
        Plotting.plotStripes(inv(self.W),plotNumbers=plotNumbers, **kwargs)

    
    def plotFilters(self, plotNumbers=False, **kwargs):
        """

        Plots the rows of the linear transform matrix W as 1D plot.

        :param plotNumbers: Determines whether the index of the basis function should be plotted as well.
        :type plotNumbers: bool
        :param **kwargs: See natter.Auxiliary.Plotting.plotStripes

        """
        Plotting.plotStripes(self.W.transpose(), plotNumbers=plotNumbers, **kwargs)


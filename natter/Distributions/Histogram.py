from __future__ import division
from Distribution import Distribution
from natter.Auxiliary.Utils import parseParameters
from numpy import array,sum,argsort,hstack,squeeze,zeros,where,cumsum,shape,linspace
from numpy.random import rand
from scipy import searchsorted, histogram
from natter.Auxiliary.Decorators import Squeezer
from natter.DataModule import Data
from matplotlib.pyplot import figure,legend,plot,show


class Histogram(Distribution):
    """
    Histogram

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.

    :param param:
        dictionary which might containt parameters for the histogram distribution
              'b'    :  bin centers

              'p'    :  probability mass for each bin center

    :type param: dict

    Primary parameters are ['p'].

    """


    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)

        # set default parameters
        self.name = 'Histogram Distribution'
        self.param = {'p':rand(2),'b':array([-1.0,1.0])}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        self.param['p'] = array(self.param['p'],dtype=float) # make sure it is a float
        self.param['p'] = self.param['p']/sum(self.param['p'])

        ind = argsort(self.param['b'])
        self.param['b'] =  self.param['b'][ind]
        self.param['p'] =  self.param['p'][ind]

        self.primary = ['p']


    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat.

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array

        '''
        b = array(self.param['b'])
        d = (b[1:] - b[:-1])/2.0
        b[1:] = b[1:]-d
        b[0] -= d[0]
        b = hstack((b,b[-1]+2.0*d[-1]))

        ind = searchsorted(b,squeeze(dat.X))
        ptmp = hstack((zeros(1),self.param['p']/(b[1:]-b[:-1]),zeros(1)))
        return ptmp[ind]

    def loglik(self,dat):
        """
        See pdf method
        """
        return self.pdf(dat)

    @Squeezer(1)
    def ppf(self,u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param u: Percentiles for which the ppf will be computed.
        :type u: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data

        '''
        P = cumsum(self.param['p'])
        ind = searchsorted(P,u)
        return Data(self.param['b'][ind],'Function values of the Histogram distribution')

    def sample(self,m):
        """

        Samples m samples from the current Histogram distribution.

        :param m: Number of samples to draw.
        :type name: int.
        :returns:  A Data object containing the samples
        :rtype:    natter.DataModule.Data

        """
        u = rand(m)
        dat = self.ppf(u)
        dat.name = "%i samples from a Histogram distribution" % (m,)
        dat.addToHistory('sampled from a Histogram distribution')
        return dat

    def histogram(self,dat,cdf = False, ax=None,plotlegend=True):
        """
        Plots a histogram of the data points in dat. This works only
        for 1-dimensional distributions. It also plots the pdf of the distribution.

        :param dat: data points that enter the histogram
        :type dat: natter.DataModule.Data
        :param cdf: boolean that indicates whether the cdf should be plotted or not (default: False)
        :type cdf: boolean
        :param ax: axes object the histogram is plotted into if it is not None.
        :type ax: matplotlib.pyplot.axis
        :param plotlegend: boolean indicating whether a legend should be plotted (default: True)
        :type plotlegens: boolean
        :param bin: number of bins to be used. If None (default), the bins are automatically determined.
        :type bin: int
        """


        b = array(self.param['b'])
        d = (b[1:] - b[:-1])/2.0
        b[1:] = b[1:]-d
        b[0] -= d[0]
        b = hstack((b,b[-1]+2.0*d[-1]))


        h = histogram(squeeze(dat.X),bins=b)[0]
        h = h/sum(h)/(b[1:]-b[:-1])
        if ax == None:
            fig = figure()
            ax = fig.add_axes([.1,.1,.8,.8])


        d2 = b[1:]-b[:-1]
        ax.bar(b[:-1],h,width=d2)

        bincenters = linspace(b[0],b[-1],1000)
        y = squeeze(self.pdf( Data(bincenters)))
        ax.plot(bincenters, y, 'k--', linewidth=2)

        if hasattr(self,'cdf') and cdf:
            z = squeeze(self.cdf( Data(bincenters)))
            ax.plot(bincenters, z, 'k.-', linewidth=2)
            if plotlegend:
                legend( ('p.d.f.','c.d.f.','Histogram') )
        elif plotlegend:
            legend( ('p.d.f.','Histogram') )

        ax.set_xlabel('x')
        ax.set_ylabel('Probability')
        ax.grid(True)


from __future__ import division
from Distribution import Distribution
from natter.Auxiliary.Utils import parseParameters
from numpy import array,sum,argsort,hstack,squeeze,zeros,where,cumsum,shape,linspace,histogram, log, amin
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
              'b'    :  bin edges
              
              'p'    :  probability mass for each bin center

    :type param: dict

    Primary parameters are ['p','b'].
        
    """


    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = parseParameters(args,kwargs)

        # set default parameters
        self.name = 'Histogram Distribution'
        self.param = {'p':None,'b':None}
        if param != None: 
            for k in param.keys():
                self.param[k] = param[k]

        self.primary = ['p','b']


    def pdf(self,dat):
        '''

        Evaluates the probability density function on the data points in dat. 

        :param dat: Data points for which the p.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the values of the density.
        :rtype:    numpy.array
           
        '''
        b = self.param['b']
        p = self.param['p']
        dt = b[1]-b[0]
        ind = searchsorted(b,squeeze(dat.X))
        
        ptmp = hstack((amin(p),self.param['p'],amin(p)))
        ptmp = ptmp/sum(ptmp)/dt
        return ptmp[ind]
    

    def loglik(self,dat):
        '''

        Computes the loglikelihood of the data points in dat.

        :param dat: Data points for which the loglikelihood will be computed.
        :type dat: natter.DataModule.Data
        :returns:  An array containing the loglikelihoods.
        :rtype:    numpy.array
         
           
        '''        
        return log(self.pdf(dat))

    def estimate(self,dat,bins=200, regularize=True):
        '''

        Estimates the parameters from the data in dat. It is possible to only 
        selectively fit parameters of the distribution by setting the primary 
        array accordingly (see :doc:`Tutorial on the Distributions module <tutorial_Distributions>`).


        :param dat: Data points on which the ExponentialPower distribution will be estimated.
        :param bins: If b was not set, bins determines the number of bins.
        :param regularize: If True, bins with no data point are set to the minimal count greater zero.
        :type dat: natter.DataModule.Data
        '''
        if 'b' in self.primary:
            p,self.param['b'] = histogram(dat.X.ravel(),bins=bins)
        else:
            p,_ = histogram(dat.X.ravel(),bins=self.param['b'])

        if regularize:
            b = self.param['b']
            dt = b[1]-b[0]
            p[p == 0] = amin(p[p>0])
        self.param['p'] = p/sum(p)/dt 
    
        
    def cdf(self,dat,nonparametric=True):
        '''
        Evaluates the cumulative distribution function on the data points in dat. 

        :param dat: Data points for which the c.d.f. will be computed.
        :type dat: natter.DataModule.Data
        :param nonparametric: Determines whether the cdf should be estimated non-parametrically. 
                  This works well if the data points in dat represent a large sample from the whole 
                  range of values. 
        :type dat: boolean
        :returns:  A numpy array containing the probabilities.
        :rtype:    numpy.array
           
        '''  
        if nonparametric:      
            u = linspace(0.,1.,dat.numex())[argsort(argsort(dat.X.ravel()))]
        else:
            b = self.param['b']
            dt = b[1]-b[0]
            P = cumsum(self.param['p'])*dt
            ind = searchsorted(b,squeeze(dat.X))
            P = hstack((0,P,1))
            u =  P[ind]
        return u


    @Squeezer(1)
    def ppf(self,u):
        '''

        Evaluates the percentile function (inverse c.d.f.) for a given array of quantiles.

        :param u: Percentiles for which the ppf will be computed.
        :type u: numpy.array
        :returns:  A Data object containing the values of the ppf.
        :rtype:    natter.DataModule.Data
           
        '''
        b = self.param['b']
        dt = b[1]-b[0]
        P = hstack((0,cumsum(self.param['p'])*dt))
        ind = searchsorted(P,u)
        return Data(b[ind],'Function values of the Histogram distribution')

    def sample(self,m):
        """

        Samples m samples from the current Histogram distribution.

        :param m: Number of samples to draw.
        :type m: int.
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
        :param ax: axes object the histogram is plotted into if it is not None.
        :param plotlegend: boolean indicating whether a legend should be plotted (default: True)
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


from Distribution import Distribution
from natter.DataModule import Data
from numpy import exp, zeros, ones, array



class Delta(Distribution):
    """
    Delta Distribution

    The constructor is either called with a dictionary, holding
    the parameters (see below) or directly with the parameter
    assignments (e.g. myDistribution(n=2,b=5)). Mixed versions are
    also possible.


    :param param:
        dictionary which might contain parameters for the uniform distribution
              'n'      :    dimensionality (default=1)
              'peak'   :    peak location (either 1D for same in all dimensions or nD)

    :type param: dict

    No primary parameters.

    """
    maxCount = 10000
    Tol = 10.0**-20.0


    def __init__(self, *args,**kwargs):
        # parse parameters correctly
        param = None
        if len(args) > 0:
            param = args[0]
        if kwargs.has_key('param'):
            if param == None:
                param = kwargs['param']
            else:
                for k,v in kwargs['param'].items():
                    param[k] = v
        if len(kwargs)>0:
            if param == None:
                param = kwargs
            else:
                for k,v in kwargs.items():
                    if k != 'param':
                        param[k] = v

        # set default parameters
        self.name = 'Delta Distribution'
        self.param = {'peak':0.0, 'n':1}
        if param != None:
            for k in param.keys():
                self.param[k] = param[k]
        self.param['peak'] =  array(self.param['peak']).reshape(self.param['n'],1)
        self.primary = []

    def sample(self,m):
        """

        Samples m samples from the delta.

        :param m: Number of samples to draw.
        :type name: int.
        :rtype: natter.DataModule.Data
        :returns:  A Data object containing the samples


        """
        return Data( ones((self.param['n'],m))*self.param['peak'], \
                     str(m) + ' samples from ' + self.name)


    def estimate(self,dat):
        '''
        Uniform distribution has no parameter hence estimate does nothing
        '''
        print self.name + ' has no parameter to fit.'


    def __setitem__(self,key,value):
        if key == 'n':
            raise NotImplementedError, 'Changing the dimensionality of ' + self.name + ' is not supported.'
        elif key == 'param':
            if array((value)).size != self.param['n']:
                raise ValueError, 'Dimensionality of delta peaks does not match dimensionality of distribution'
            self.param['peak'] =  self.param['peak'].reshape(self.param['n'],1)
        self.param[key] = value


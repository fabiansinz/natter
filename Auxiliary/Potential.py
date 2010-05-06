from __future__ import division
import Data
from numpy import sign, exp, abs

class Potential:



    def __init__(self,ptype='studentt',params=None):
        """

        POTENTIAL(PTYPE,PARAMS=None)

        defines a potential class used e.g. by the Product of Experts model. Possible ptypes are:

        \"studentt\" : potential of the form exp(-|x|), since there are no parameters param=None
        
        """
        self.ptype = ptype
        self.params = None


    def dlogdx(self,dat):
        """
        Returns the derivative of the log of the potential w.r.t to the data points in dat.
        """

        if self.ptype == 'laplace':
            return -sign(dat.X)
        elif self.ptype == 'studentt':
            return -2.0*dat.X / (1.0 + dat.X**2.0)




    def d2logdx2(self,dat):
        """
        Returns the second derivative of the log of the potential w.r.t to the data points in dat.
        """
        if self.ptype == 'laplace':
            return 0.0*dat.X
        elif self.ptype == 'studentt':
            return -2.0 *(1.0-dat.X**2.0) / (1.0 + dat.X**2)**2



    def d3logdx3(self,dat):
        """
        Returns the third derivative of the log of the potential w.r.t to the data points in dat.
        """

        if self.ptype == 'laplace':
            return 0.0*dat.X
        elif self.ptype == 'studentt':
            return 4.0*dat.X* ( (1.0+dat.X**2)**-2.0 + 2.0 * (1.0 - dat.X**2.0)/ (1.0 + dat.X**2)**3.0)


    def __call__(self,dat):
        if self.ptype == 'laplace':
            return exp(-abs(dat.X))
        elif self.ptype == 'studentt':
            return (1.0 + dat.X**2)**-1.0


    def __repr(self):
        return self.__str__()

    def __str__(self):
        return self.ptype + ' potential'
    

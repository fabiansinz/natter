import Transform
import LinearTransform
import numpy as np
import string
from natter.Auxiliary import Errors
from natter.DataModule import Data
import types
import copy
from copy import deepcopy
from natter.Auxiliary.Utils import displayHistoryRec

class NonlinearTransform(Transform.Transform):
    '''
    NonlinearTransform class.

    :param f: Function on data representing the mapping for this NonlinearTransform object.
    :type f: function
    :param name: Name of the NonlinearTransform object.
    :type name: string
    :param history: History of the object.
    :type history: List of (list of ...) strings
    :param logdetJ: Function that computes the log-det-Jacobian of the NonlinearTransform on data.
    :type logdetJ: function
    '''

    def __init__(self,f=None,name='Noname',history=None,logdetJ=None):
        if history == None:
            self.history = []
        else:
            self.history = history
        self.f = f
        self.name = name
        self.logdetJ = logdetJ


    def apply(self,O):
        """
        Applies the NonlinearTransform object to *O*. *O* can either be

        * a natter.Transforms.LinearTransform object
        * a natter.Transforms.NonlinearTransform object
        * a natter.DataModule.Data object

        It also updates the correct computation of the log-det-Jacobian.
        
        :param O: Object this NonlinearTransform is to be applied to.
        :type O: see above.
        :returns: A new Transform or Data object
        :rtype: Depends on the type of *O*
        
        """
        
        if isinstance(O,Data):
            # copy other history 
            tmp = deepcopy(O.history)
            tmp.append('Applied non-linear filter "' + self.name + '"')
            tmp.append(deepcopy(self.history))

            # compute results and add own history
            ret = self.f(O)
            ret.history.append(tmp)
            return ret
        elif isinstance(O,LinearTransform.LinearTransform):
            # copy other history and add own
            tmp = deepcopy(O.history)
            tmp.append('multiplied with Transform "' + self.name + '"')
            tmp.append(deepcopy(self.history))

            Ocpy = O.copy()
            Scpy = self.copy()
            g = lambda x: Scpy.f(Ocpy.f(x))
            gdet = None
            if Scpy.logdetJ != None:
                gdet = lambda y: self.logdetJ(Ocpy.apply(y)) + Ocpy.logDetJacobian()
            return NonlinearTransform(g,O.name,tmp, logdetJ=gdet )
        elif isinstance(O,NonlinearTransform):
            # copy other history and add own
            tmp = deepcopy(O.history)
            tmp.append('composed with "' + self.name + '"')
            tmp.append(deepcopy(self.history))

            Scpy = self.copy()
            Ocpy = O.copy()
            g = lambda x: Scpy.f( Ocpy.f(x) )
            gdet = None
            if self.logdetJ != None and O.logdetJ != None:
                gdet = lambda y: Scpy.logdetJ(Ocpy.f(y)) + Ocpy.logdetJ(y)
            return NonlinearTransform(g,O.name,tmp, logdetJ=gdet )
        else:
            raise TypeError('Transform.NoninearTransform.__mult__(): Transforms can only be multiplied with Data, Transform.LinearTransform or Transform.NonlinearTransform objects')
        return self


    def logDetJacobian(self,dat):
        """
        Computes the determinant of the logarithm of the Jacobians
        determinant for the nonliner transformation at each data point
        in *dat*.


        :param dat: Data for which the log-det-Jacobian is to be computed.
        :type dat: natter.DataModule.Data
        :returns: The log-det-Jacobian 
        :rtype: numpy.array
        """
        
        if self.logdetJ == None:
            raise Errors.AbstractError('logdetJ has not been specified!')
        else:
            return self.logdetJ(dat)

    def __call__(self,O):
        """
        The same as apply(O). Overloades the call operator.

        :param O: Object this NonlinearTransform is to be applied to.
        :type O: see above.
        :returns: A new Transform or Data object
        :rtype: Depends on the type of *O*
        """
        return self.apply(O)

    def __str__(self):
        """
        Returns a string representation of the NonlinearTransform object.

        :returns: A string representation of the NonlinearTransform object.
        :rtype: string
        """
        
                
        s = 30*'-'
        s += '\nNonlinear Transform: ' + self.name + '\n'
        if len(self.history) > 0:
            s += displayHistoryRec(self.history)
        s += 30*'-'
        
        return s

    def html(self):
        """
        Returns an html representation of itself. This is required by
        LogToken which LinearTransform inherits from.

        :returns: html preprentation the LinearTransform object
        :rtype: string
        """
        s = "<table border=\"0\"rules=\"groups\" frame=\"box\">\n"
        s += "<thead><tr><td colspan=\"2\"><tt><b>Nonlinear Transform: %s</b></tt></td></tr></thead>\n" % (self.name,)
        s += "<tbody>"
        s += "<tr><td valign=\"top\"><tt>History: </tt></td><td><pre>"
        if len(self.history) > 0:
            s += displayHistoryRec(self.history)
        s += "</pre></td></tr></table>"
        return s

    def getHistory(self):
        """
        Returns the history of the object. The history is a list of
        (list of ...) strings that store the previous operations
        carried out on the object.

        :returns: The history.
        :rtype: list of (list of ...) strings
        """
        
        return deepcopy(self.history)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        """
        Makes a deep copy of the NonlinearTransform and returns it.

        :returns: A deep copy of the NonlinearTransform object.
        :rtype: natter.Transforms.NonlinearTransform
        """
        
        return copy.deepcopy(self)

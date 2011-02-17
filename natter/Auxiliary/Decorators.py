import types
from numpy import where, any
from warnings import warn

class DataSupportChecker:
    """
    Decorator that checks whether the data passed to a class function
    is in a certain range.

    The constructor takes three arguments, one that specifies the
    index of the data object in the parameter list of the decorated
    function. The other two specify the lower and the upper
    boundary. The boundaries can either be strings, in which case they
    have to be keys into param of the distribution, or they can be
    float, in which case they are used as they are.

    If the data does not comply with the specified range, a warning is
    raised and the data is set appropriately.

    :param nArg: index of the Data object in the parameter list of the decorated function
    :type nArg: int
    :param lb: lower bound
    :type lb: str or int
    :param ub: upper bound
    :type ub: str or int
    
    """
    def __init__(self, nArg, lb, ub):
        self.nArg = nArg
        self.lb = lb
        self.ub = ub
        
    def __call__(self, f):
        def wrapped_f(*args):
            if type(self.lb) == types.StringType:
                self.lb = args[0].param[self.lb]
            if type(self.ub) == types.StringType:
                self.ub = args[0].param[self.ub]
            if any(args[self.nArg].X < self.lb):
                warn("Data is outside the distributions support. Setting data points to support boundary.")
                args[self.nArg].X[where(args[self.nArg].X < self.lb)] = self.lb 
            if any(args[self.nArg].X > self.ub):
                warn("Data is outside the distributions support. Setting data points to support boundary.")
                args[self.nArg].X[where(args[self.nArg].X > self.ub)] = self.ub 

            return f(*args)
        return wrapped_f




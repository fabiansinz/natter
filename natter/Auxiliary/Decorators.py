import types
from numpy import where, any,squeeze, sum
from warnings import warn

# class DataSupportChecker:
#     """
#     Decorator that checks whether the data passed to a class function
#     is in a certain range.

#     The constructor takes three arguments, one that specifies the
#     index of the data object in the parameter list of the decorated
#     function. The other two specify the lower and the upper
#     boundary. The boundaries can either be strings, in which case they
#     have to be keys into param of the distribution, or they can be
#     float, in which case they are used as they are.

#     If the data does not comply with the specified range, a warning is
#     raised and the data is set appropriately.

#     :param nArg: index of the Data object in the parameter list of the decorated function
#     :type nArg: int
#     :param lb: lower bound
#     :type lb: str or float
#     :param ub: upper bound
#     :type ub: str or float
    
#     """
#     def __init__(self, nArg, lb, ub):
#         self.nArg = nArg
#         self.lb = lb
#         self.ub = ub
        
#     def __call__(self, f):
#         def wrapped_f(*args):
#             if type(self.lb) == types.StringType:
#                 self.lb = args[0].param[self.lb]
#             if type(self.ub) == types.StringType:
#                 self.ub = args[0].param[self.ub]


#             print "DataSupportChecker: lb=%.2f and ub=%.2f" % (self.lb,self.ub)

#             if any(args[self.nArg].X < self.lb):
#                 # print args[self.nArg].X[where(args[self.nArg].X<self.lb)]
#                 # raw_input()
                
#                 warn("Data is outside the distributions support. Setting %i data points to support boundary." % (sum( (args[self.nArg].X < self.lb).flatten() ),))
#                 args[self.nArg].X[where(args[self.nArg].X < self.lb)] = self.lb
#             if any(args[self.nArg].X > self.ub):
#                 # print args[self.nArg].X[where(args[self.nArg].X > self.ub)]
#                 # raw_input()
#                 warn("Data is outside the distributions support. Setting %i data points to support boundary."% (sum( (args[self.nArg].X > self.ub).flatten() ),))

#                 args[self.nArg].X[where(args[self.nArg].X > self.ub)] = self.ub

#             return f(*args)
#         return wrapped_f



class ArraySupportChecker:
    """
    Decorator that checks whether the numpy.arrat passed to a class function
    is in a certain range.

    The constructor takes three arguments, one that specifies the
    index of the data object in the parameter list of the decorated
    function. The other two specify the lower and the upper
    boundary. The boundaries must be floats.

    If the data does not comply with the specified range, a warning is
    raised and the data is set appropriately.

    :param nArg: index of the Data object in the parameter list of the decorated function
    :type nArg: int
    :param lb: lower bound
    :type lb: float
    :param ub: upper bound
    :type ub: float
    
    """
    def __init__(self, nArg, lb, ub):
        self.nArg = nArg
        self.lb = lb
        self.ub = ub
        
    def __call__(self, f):
        def wrapped_f(*args):
            if any(args[self.nArg] < self.lb):
                warn("Array is outside the distributions support. Setting %i data points to support boundary." % (sum( (args[self.nArg] < self.lb).flatten() ),))
                args[self.nArg][where(args[self.nArg] < self.lb)] = self.lb
            if any(args[self.nArg] > self.ub):
                warn("Array is outside the distributions support. Setting %i data points to support boundary."% (sum( (args[self.nArg] > self.ub).flatten() ),))
                args[self.nArg][where(args[self.nArg] > self.ub)] = self.ub

            return f(*args)
        return wrapped_f

def OutputRangeChecker(a,b):
    """
    Decorator that makes sure the output array is in [a,b].

    :param a: lower bound
    :type a: float
    :param b: upper bound
    :type b: float
    
    """
    def wrap(f):
        def wrapped_f(*args):
            out = f(*args)
            if any(out < a):
                warn('There are some output lower than %.2f. Setting them to %.2f' % (a,a))
                out[where(out < a)] = a
            if any(out > b):
                warn('There are some output greater than %.2f. Setting them to %.2f' % (b,b))
                out[where(out > b)] = b
            
            return out
        return wrapped_f
    return wrap
    

def Squeezer(n):
    """
    Decorator that squeezes the nth argument.

    :param n: number of the argument to be squeezed
    :type n: int
    
    """
    def wrap(f):
        def wrapped_f(*args):
            args = list(args)
            args[n] = squeeze(args[n])
            return f(*args)
        return wrapped_f
    return wrap
 

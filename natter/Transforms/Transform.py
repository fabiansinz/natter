import cPickle as pickle
from natter import Auxiliary
import types
from natter.Logging.LogTokens import LogToken

class Transform(LogToken):
    '''
    Mother class of all transformations.

    '''

    def __init__(self):
        raise Auxiliary.Errors.AbstractError('Filter is only an abstract class!')

    def logDetJacobian(self):
        """
        Abstract method for the computation of the log-det-Jacobian
        which is to be implemented by the children that inherit from
        Transform.
        """
        raise Auxiliary.Errors.AbstractError('Abstract method logDetJacobian() not implemented in ' + self.name)
        

    def addToHistory(self,hi):
        """
        Appends an item to the history of the transformation.

        :param hi: Item to be added to the history.
        :type hi: string or list of (list of ...) strings
        """
        self.history.append(hi)

    def __mul__(self,O):
        """
        Overloads the multiplication operator. This is equivalent to call self.apply(O)

        :param O: Object *apply* is to be called on.
        :type O: Transform object or a Data object.
        :returns: A new Transform object
        :rtype: natter.Transforms.Transform
        """
        return self.apply(O)

    def apply(self):
        """
        Abstract method for the application of the Transform object
        which is to be implemented by the children that inherit from
        Transform.
        """
        raise Auxiliary.Errors.AbstractError('Abstract method apply() not implemented in ' + self.name)

            
    def save(self,filename):
        """
        Save the filter object to the specified file.

        :param filename: Filename for the save file.
        :type filename: string
        """
        Auxiliary.save(self,filename)

    def ascii(self):
        """
        Returns an ascii representation of itself. This is required by
        LogToken which Transform inherits from.

        :returns: ascii preprentation the Transform object
        :rtype: string
        """
        
        return self.__str__()

        



def load(path):
    """
    Loads a saved Transform object from the specified path.

    :param path: Path to the saved Transform object.
    :type path: string
    :returns: The loaded object.
    :rtype: natter.Transforms.Transform
    """
    f = open(path,'rb')
    ret = pickle.load(f)
    f.close()
    return ret

def displayHistoryRec(h,recDepth=0):
    s = ""
    for elem in h:
        if type(elem) == types.ListType:
            s += (recDepth-1)*'   ' +  displayHistoryRec(elem,recDepth+1)
        else:
            s += recDepth*'   ' + ' |-' + elem + '\n'
    return s

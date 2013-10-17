
class InitializationError(Exception):
    """
    Error that is raised if parameters are not properly initialized.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class AbstractError(Exception):
    """
    Error that is raised if inherited method is not implemented.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class DimensionalityError(Exception):
    """
    Error that is raised if dimensions do not fit.
    """
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class UpdateError(Exception):
    """
    Error that is raised if updates in a learning process fail.
    """
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class DataLoadingError(Exception):
    """
    Error that is raised if data cannot be loaded.
    """
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class SpecificationError(Exception):
    """
    Error that is raised if parameters are not specified properly.
    """
    def __init__(self,value):
        Exception.__init__(self)



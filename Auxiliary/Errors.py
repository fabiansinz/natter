import types



class ValueError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
    

class AbstractError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TestError(Exception):
    def __init__(self,value=None):
        if value == None:
            self.value = {'analytics':'- no analytics specified - '}
        elif type(value) == types.StringType:
            self.value = {'message':value}
        else:
            self.value = value

    def __str__(self):
        s = "\n"
        s+= "++++++++++++++++++++++++ Test Error Protocol ++++++++++++++++++++++++\n"
        for (k,v) in self.value.items():
            s += str(k).upper()  + ": \n"
            s += str(v) + '\n'
            s += 10*'- - ' + '\n'
        s += "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        return s

    def __repr__(self):
        return self.__str__()

class DimensionalityError(Exception):
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class UpdateError(Exception):
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class DataLoadingError(Exception):
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)


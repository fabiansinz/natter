import types


class ExperimentProtocol:

    _HTML_HEADER = """
                   <!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">
                   <html>
                   <head>
                   <meta content=\"text/html; charset=ISO-8859-1\"
                   http-equiv=\"Content-Type\">
                   <title>%s</title>
                   </head>\n
                   """

    _HTML_FOOTER = """
                   \n</body>
                   </html>
                   """

    def __init__(self,name="New Experiment Protocol", sections=()):
        self._sections = sections # sections of the protocol
        self._subprotocols = {}
        self._text = []
        for k in self._sections:
            self._subprotocols[k] = ExperimentProtocol(k)
        self._name = name # name of the protocol

    def addSection(self,section,val=None):
        if not section in self._sections:
            self._sections += (section,)
            if val == None:
                self._subprotocols[section] = ExperimentProtocol(section)
            else:
                self._subprotocols[section] = val
        else:
            raise KeyError("Section %s already exists" % (section,))

    def __add__(self,item):
        if type(item) != types.StringType:
            raise TypeError('Only Strings can be added to the protocol')
        else:
            self._text.append(item)
        return self
        
    def __getitem__(self,k):
        if k in self._sections:
            return self._subprotocols[k]
        else:
            raise KeyError("Section %s does not exist" % (k,))

    def __setitem__(self,k,value):
        if isinstance(value,ExperimentProtocol):
            self._subprotocols[k] = value
        else:
            raise TypeError("Assignment value must be a ExperimentProtocol")


    def __str__(self):
        s = 50*"=" + "\n"
        s += "Experiment Protocol %s:\n" % (self._name,)
        s += 50*"-" + "\n"
        s += "\n\n".join(self._text) + "\n"
        s += 50*"=" + "\n"

        return s

    def __repr__(self):
        return self.__str__()
        
    def write(self,filename,format='html'):
        with open(filename,'w') as f:
            f.write(self._HTML_HEADER % (self._name,))
            
            f.write(self._HTML_FOOTER)

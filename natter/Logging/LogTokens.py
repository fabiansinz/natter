from natter.Auxiliary.Errors import AbstractError
from Utils import lrfill, hLine
import types


class LogToken:

    def latex(self):
        raise AbstractError("Method latex not implemented in this ProtocolToken!")

    def html(self):
        raise AbstractError("Method html not implemented in this ProtocolToken!")

    def wiki(self):
        raise AbstractError("Method wiki not implemented in this ProtocolToken!")

    def ascii(self):
        raise AbstractError("Method ascii not implemented in this ProtocolToken!")

    def __log__(self,type='ascii'):
        return getattr(self,type)()        


############################################################################
class Table(LogToken):

    
    def __init__(self,rows = (), cols = ()):
        self._content = {}
        self._cols = cols
        self._rows = rows
        self.FLOAT_FORMAT = "%.4g"
        for rk in rows:
            self._content[rk] = {}
            for ck in cols:
                self._content[rk][ck] = ""


    def __getitem__(self,k):
        if k[0] in self._rows and k[1] in self._cols:
            return self._content[k[0]][k[1]]
        else:
            raise KeyError("Key (%s,%s) does not exist" % (str(k[0]),str(k[1])))

    def __setitem__(self,k,val):
        if k[0] in self._rows and k[1] in self._cols:
            self._content[k[0]][k[1]] = val
        else:
            raise KeyError("Key (%s,%s) does not exist" % (str(k[0]),str(k[1])))

    def __str__(self):
        return self.ascii()

    def ascii(self):
        n = 0
        for ck in self._cols:
            if len(str(ck)) > n:
                n = len(str(ck))

        for rk in self._rows:
            if len(str(rk)) > n:
                n = len(str(rk))
            for ck in self._cols:
                if type(self._content[rk][ck]) == types.FloatType:
                    m =  len(self.FLOAT_FORMAT % (self._content[rk][ck],))
                elif type(self._content[rk][ck]) == types.StringType:
                    m =  len(self._content[rk][ck])
                else:
                    raise TypeError("Data type of (%s,%s) not known" % (str(rk),str(ck)))
                    
                if m > n:
                    n = m

        n +=2

        #-------
        cm = len(self._cols) + 1
        ret = hLine(cm,n) + "\n"
        ret += "|" + n*" " + "|" + "|".join([lrfill(elem,n) for elem in self._cols]) + "|\n"
        ret += hLine(cm,n) + "\n"
        
        
        for rk in self._rows:
            row = [str(rk)]
            for ck in self._cols:
                if type(self._content[rk][ck]) == types.FloatType:
                    row.append(self.FLOAT_FORMAT % (self._content[rk][ck],))
                elif type(self._content[rk][ck]) == types.StringType:
                    row.append(self._content[rk][ck])
                else:
                    raise TypeError("Data type of (%s,%s) not known" % (str(rk),str(ck)))
            ret += "|" + "|".join([lrfill(elem,n) for elem in row]) + "|\n"
            ret += hLine(cm,n) + "\n"
        return ret

##############################################

class Paragraph(LogToken):

    def ascii(self):
        return "\n\n"

    def wiki(self):
        return "\n\n"
    
    def html(self):
        return "<p>"

    def latex(self):
        return "\n\n"

##############################################

class Link(LogToken):

    def __init__(self,target, name=None):
        self._target = target
        if name == None:
            self._name = target
        else:
            self._name = name
        
    def ascii(self):
        return self._name

    def wiki(self):
        return "[%s %s]" % (self._target,self._name)
    
    def html(self):
        return "<a href=\"%s\">%s</a>" % (self._target,self._name)

    def latex(self):
        return self._name

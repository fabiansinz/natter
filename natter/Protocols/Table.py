from Utils import lrfill, hLine

import types

class Table:

    
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
        return self._content[k[0]][k[1]]

    def __setitem__(self,k,val):
        self._content[k[0]][k[1]] = val

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
    

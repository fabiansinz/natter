from natter.Auxiliary.Errors import AbstractError
from Utils import lrfill, hLine
import types
import textwrap
import re
import os
from numpy import float64, float32
from subprocess import Popen, PIPE
from os import path, chdir
from time import time, strftime, localtime
import sys
import numpy
import scipy

class LogToken:
    """
    Abstract class each LogToken inherits from. Forces LogTokens to
    implement the methods *latex*, *ascii*, *html* and *wiki*.
    """
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



##############################################
class PyInfo(LogToken):
    """
    Represents information about the state of a python.
    """

    
    def __init__(self):
        info = {}
        info['Python Version'] = sys.version
        info['Numpy Version'] = numpy.version.version
        info['Scipy Version'] = scipy.version.version
        
        
        self.info = info
        
    def ascii(self):
        """
        :returns: An ascii representation of the python info.
        :rtype: string
        """

        return "\n".join(["%s: %s" % (k,v) for k,v in self.info.items()])


    
    def html(self):
        """
        :returns: A html representation of the python info.
        :rtype: string
        """
        return "<table border=\"0\"><tr>" + "</tr><tr>".join(["<td>%s</td><td> %s</td>" % (k,v) for k,v in self.info.items()]) + "</td></table>"
##############################################

class Git(LogToken):
    """
    Represents information about the state of a git directory.
    """

    
    def __init__(self,gitdir='./'):
        self.gitdir = gitdir
        self.boolPretty = {True:'yes', False:'no'}
        olddir = os.path.abspath('./')
        chdir(self.gitdir)
        # get commit hash
        pr1 = Popen(['git', 'log', '-1'], stdout=PIPE)
        pr2 = Popen(['head', '-1'], stdin=pr1.stdout, stdout=PIPE)
        pr3 = Popen(['cut', '-d', ' ', '-f', '2'], stdin=pr2.stdout, stdout=PIPE)
        self.commit = pr3.communicate()[0][:-1]
        
        # check if project contains uncommitted changes
        pr1 = Popen(['git', 'status', '--porcelain'], stdout=PIPE)
        pr2 = Popen(['egrep', '^.M'], stdin=pr1.stdout, stdout=PIPE)
				
        if pr2.communicate()[0]:
            self.modified = True
        else:
            self.modified = False
        chdir(olddir)

    def ascii(self):
        """
        :returns: An ascii representation of the Git Repository.
        :rtype: string
        """

        return "Git directory: %s\ncommit: %s\nuncommited changes: %s" % (self.gitdir, self.commit, self.boolPretty[self.modified])

    
    def html(self):
        """
        :returns: A html representation of the git directory.
        :rtype: string
        """
        return "<table border=\"0\"><tr><td>Git directory:</td><td>%s</td></tr><tr><td>commit:</td><td> %s</td></tr><tr><td>uncommited changes: </td><td>%s</td></tr></table>" % (self.gitdir, self.commit, self.boolPretty[self.modified])


##############################################

class Svn(LogToken):
    """
    Represents information about the state of a svn directory.
    """

    
    def __init__(self,svndir='./'):
        self.boolPretty = {True:'yes', False:'no'}
        olddir = os.path.abspath('./')
        chdir(svndir)
        # get commit hash
        pr1 = Popen(['svn', 'info'], stdout=PIPE)
        self.svninfo = pr1.stdout.readlines()
        chdir(olddir)

    def ascii(self):
        """
        :returns: An ascii representation of the Git Repository.
        :rtype: string
        """

        return "SVN information:\n %s\n" % (''.join(self.svninfo),)

    
    def html(self):
        """
        :returns: A html representation of the git directory.
        :rtype: string
        """
        return "SVN information <br><table border=\"0\"><tr><td>" + '</td></tr><tr><td>'.join(self.svninfo) + '</td></tr></table>'

############################################################################
class Table(LogToken):
    """
    Table LogToken to store results. Each table is initialized with
    two tuples: the row labels and the column labels. Table cells can
    be accesses via these labels. For examples

    >>> t = Table((1,2),('a','b'))
    >>> t[1,'b'] = 1.2

    Cell contents can be floats, strings or LogTokens again. 
    """
    
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
        """
        :returns: An ascii representation of the Table.
        :rtype: string
        """
        n = 0
        for ck in self._cols:
            if len(str(ck)) > n:
                n = len(str(ck))

        for rk in self._rows:
            if len(str(rk)) > n:
                n = len(str(rk))
            for ck in self._cols:
                if type(self._content[rk][ck]) == types.FloatType  or type(self._content[rk][ck]) == float64 or  type(self._content[rk][ck]) == float32:
                    m =  len(self.FLOAT_FORMAT % (self._content[rk][ck],))
                elif type(self._content[rk][ck]) == types.StringType:
                    m =  len(self._content[rk][ck])
                elif isinstance(self._content[rk][ck],LogToken):
                    m = len(self._content[rk][ck].ascii())
                else:
                    raise TypeError("Data type of (%s,%s) not known" % (str(rk),str(ck)))
                    
                if m > n:
                    n = m

        n +=2

        #-------
        cm = len(self._cols) + 1
        ret = hLine(cm,n) + "\n"
        ret += "|" + n*" " + "|" + "|".join([lrfill(str(elem),n) for elem in self._cols]) + "|\n"
        ret += hLine(cm,n) + "\n"
        
        
        for rk in self._rows:
            row = [str(rk)]
            for ck in self._cols:
                if type(self._content[rk][ck]) == types.FloatType or type(self._content[rk][ck]) == float64 or  type(self._content[rk][ck]) == float32:
                    row.append(self.FLOAT_FORMAT % (self._content[rk][ck],))
                elif type(self._content[rk][ck]) == types.StringType:
                    row.append(self._content[rk][ck])
                elif isinstance(self._content[rk][ck],LogToken):
                    row.append(self._content[rk][ck].ascii())
                else:
                    raise TypeError("Data type of (%s,%s) not known" % (str(rk),str(ck)))
            ret += "|" + "|".join([lrfill(elem,n) for elem in row]) + "|\n"
            ret += hLine(cm,n) + "\n"
        return ret

    def html(self):
        """
        :returns: A html representation of the table.
        :rtype: string
        """
        b = False
        s = "<br><table style=\"background-color: rgb(200, 200, 200);\" border=\"1\" cellpadding=\"8\" cellspacing=\"3\" rules=\"cols\" border=\"0\" rules=\"rows\">"
        s += "<tr><td></td>"
        for colelem in self._cols:
            s += "<td><b>%s</b></td>" % (str(colelem),)
    
        s += "</tr>"

        for rk in self._rows:
            if b:
                s += "<tr style=\"background-color: rgb(221, 221, 221);\"><td><b>%s</b></td>" % (str(rk),)
                b = False
            else:
                s += "<tr style=\"background-color: rgb(255, 255, 255);\"><td><b>%s</b></td>" % (str(rk),)
                b = True
            for ck in self._cols:
                s += "<td>"
                if type(self._content[rk][ck]) == types.FloatType or type(self._content[rk][ck]) == float64 or  type(self._content[rk][ck]) == float32:
                    s += self.FLOAT_FORMAT % (self._content[rk][ck],)
                elif type(self._content[rk][ck]) == types.StringType:
                    s += self._content[rk][ck]
                elif isinstance(self._content[rk][ck],LogToken):
                    s += self._content[rk][ck].html()
                    
                s += "</td>"

            s += "</tr>"

        s+= "</table><br>"
        return s

    def latex(self):
        """
        :returns: A latex representation of the table.
        :rtype: string
        """
        
        s = "\\begin{center}\\begin{tabular}{|%s|}\\hline" % ("|".join((len(self._cols)+1)*['l']))
        s += "&" + "&".join([str(elem) for elem in self._cols]) + "\\\\ \\hline\\hline"
    

        for rk in self._rows:
            s += "\\bf %s " % (str(rk),)
            for ck in self._cols:
                s += "&"
                if type(self._content[rk][ck]) == types.FloatType or type(self._content[rk][ck]) == float64 or  type(self._content[rk][ck]) == float32:
                    s += self.FLOAT_FORMAT % (self._content[rk][ck],)
                elif type(self._content[rk][ck]) == types.StringType:
                    s += self._content[rk][ck]
                elif isinstance(self._content[rk][ck],LogToken):
                    s += self._content[rk][ck].latex()
                    

            s += "\\\\\\hline"

        s+= "\\end{tabular}\\end{center}"
        return s
        

##############################################

class Paragraph(LogToken):
    """
    Represents a paragraph.
    """

    def ascii(self):
        """
        :returns: An ascii representation of the Paragraph.
        :rtype: string
        """
        return "\n\n"

    def wiki(self):
        """
        :returns: A wiki representation of the paragraph.
        :rtype: string
        """
        
        return "\n\n"
    
    def html(self):
        """
        :returns: A html representation of the paragraph.
        :rtype: string
        """
        return "<p>"

    def latex(self):
        """
        :returns: A latex representation of the paragraph.
        :rtype: string
        """
        return "\n\n"

##############################################

class Link(LogToken):
    """
    Represents links: Each link can either be initialized with its
    target only or with an additional link name. For example:

    >>> l = Link('myfile.html')
    >>> l2 = Link('myfile.html','This is a link to a file')
    
    """

    def __init__(self,target, name=None):
        self._target = target
        if name == None:
            self._name = target
        else:
            self._name = name
        
    def ascii(self):
        """
        :returns: An ascii representation of the link.
        :rtype: string
        """
        return self._name

    def wiki(self):
        """
        :returns: An wiki representation of the link.
        :rtype: string
        """
        return "[%s %s]" % (self._target,self._name)
    
    def html(self):
        """
        :returns: An html representation of the link.
        :rtype: string
        """
        return "<a href=\"%s\">%s</a>" % (self._target,self._name)

    def latex(self):
        """
        :returns: An latex representation of the link.
        :rtype: string
        """
        return "%s" % (self._name,)

##############################################

class Image(LogToken):
    """
    Represents images: Each image is initialized with its
    filename. For example:

    >>> I = Image('whatever.png')
    
    """

    def __init__(self,name):
        self._name = name
        
    def ascii(self):
        """
        :returns: An ascii representation of the image.
        :rtype: string
        """
        return self._name

    def wiki(self):
        """
        :returns: An wiki representation of the image.
        :rtype: string
        """
        return "%s" % (self._name,)
    
    def html(self):
        """
        :returns: An html representation of the image.
        :rtype: string
        """
        return "<img src=\"%s\">" % (self._name,)

    def latex(self):
        """
        :returns: An latex representation of the image.
        :rtype: string
        """
        return "\\includegraphics{%s}" % (self._name,)


##########################################################

class LogList(LogToken):
    """
    LogList implements a list of strings or LogTokens.

    At initialization the type can be set to *item* in which case the
    enumeration is unnumbered or to *arabic*, in which case the list is
    enumerated with arabic numbers.

    :param etype: Enumeration format. Possible values are: items, arabic
    :type etype: string 
    """

    def __init__(self,etype='items'):
        self._type = etype
        self._list = []

    def __add__(self,val):
        if type(val) != types.StringType and not isinstance(val,LogToken):
            raise TypeError("List elements must be strings or LogTokens")
        else:
            if type(val) == types.StringType:
                self._list.append(re.sub("\s+", " ", val))
            else:
                self._list.append(val)
        return self

    def ascii(self):
        """
        :returns: An ascii representation of the list.
        :rtype: string
        """
        s = "\n"
        counter = 1
        for elem in self._list:
            symbol = ""
            if self._type == 'arabic':
                symbol = str(counter) + ". "
                counter += 1
            elif self._type == 'items':
                symbol = "* "
                
            if type(elem) == types.StringType:
                s += "%s%s\n" % (symbol,textwrap.fill(elem,80))
            elif isinstance(elem,LogToken):
                s += "%s%s\n" % (symbol,elem.ascii())
        return s
    
    def html(self):
        """
        :returns: A html representation of the list.
        :rtype: string
        """
        s = "<br>"
        if self._type == "items":
            s += "<ul>"
        elif self._type == 'arabic':
            s += "<ol>"
            
        for elem in self._list:
            if type(elem) == types.StringType:
                s += "<li>%s</li>\n" % (elem, )
            elif isinstance(elem,LogToken):
                s += "<li>%s</li>\n" % (elem.html(), )
                
        if self._type == "items":
            s += "</ul>"
        elif self._type == 'arabic':
            s += "</ol>"
            
        return s + "<br>"

    def latex(self):
        """
        :returns: A latex representation of the list.
        :rtype: string
        """
        s = ""
        if self._type == "items":
            s += "\\being{itemize}"
        elif self._type == 'arabic':
            s += "\\begin{enumerate}"
            
        for elem in self._list:
            if type(elem) == types.StringType:
                s += "\\item %s\n" % (elem, )
            elif isinstance(elem,LogToken):
                s += "\\item %s\n" % (elem.html(), )
                
            
        if self._type == "items":
            return s + "\\end{itemize}"
        elif self._type == 'arabic':
            return s + "\\end{enumerate}"


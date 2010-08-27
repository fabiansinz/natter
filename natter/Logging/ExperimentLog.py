import types
from LogTokens import LogToken, Link, Paragraph
import textwrap
import re
class ExperimentLog(LogToken):
    """
    Class for storing experiment logs. ExperimentLog is a :doc:`LogToken <Logging_LogTokens>`.

    ExperimentLog overloads the +, the / and the * operator. + and /
    can be used to add new content to the log. They accept strings and
    LogTokens. The only difference between them is that / adds a line
    break. * adds links to the experiment logs. Links can be specified
    by the path (string) only or by a tuple of two strings of which
    the first denotes the path and the second denotes the link
    name. For example

    >>> p = ExperimentLog('My fancy experiment')
    >>> p += 'We sampled of data we found on the website:'
    >>> p *= ('http://dataparadise.com','data paradise')

    ExperimentLog allows for adding subsections with the function
    *addSection*. These sections can be accessed under the section
    name like in a dictionary. Each section is a ExperimentLog itself.

    >>> p.addSection('Results')
    >>> p['Results'] += 'The following section summarizes our results!'
    
    A new experiment log is initialized with an empty parameter list or the name of the log.

    :param name: Name of the experiment log
    :type name: string

    """

    _HTML_HEADER = """
                   <!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">
                   <html>
                   <head>
                   <meta content=\"text/html; charset=ISO-8859-1\"
                   http-equiv=\"Content-Type\">
                   <title>%s</title>
                   </head><body>\n
                   """

    _HTML_FOOTER = """
                   \n</body>
                   </html>
                   """

    _ASCII_HEADER = 80*"=" + "\n\nExperiment Log \"%s\":\n\n" + 80*"-" + "\n\n"
    _ASCII_FOOTER = "\n\n" + 80*"="

    def __init__(self,name="New Experiment Log", sections=()):
        self._sections = sections # sections of the log
        self._sublogs = {}
        self._log = []
        for k in self._sections:
            self._sublogs[k] = ExperimentLog(k)
        self._name = name # name of the log

    def addSection(self,section,val=None):
        """
        Adds a new subsection to the log.

        :param section: The subsection's name
        :type section: string
        :param val: Assign a experiment log to the subsection.
        :type val: natter.Logging.ExperimentLog
        """
        if not section in self._sections:
            self._sections += (section,)
            if val == None:
                self._sublogs[section] = ExperimentLog(section)
            else:
                self._sublogs[section] = val
        else:
            raise KeyError("Section %s already exists" % (section,))

    def __add__(self,item):
        if type(item) == types.ListType:
            for elem in item:
                self += elem
        if type(item) != types.StringType and not isinstance(item,LogToken):
            raise TypeError('Only Strings, LogTokens or lits of both can be added to the log')
        else:
            if type(item) == types.StringType:
                item = re.sub("\s+", " ", item)
            self._log.append(item)
        return self

    def __div__(self,item):
        self += Paragraph()
        return self + item

    def __mul__(self,item):
        if type(item) == types.StringType:
            self += Link(item)
        elif type(item) == types.TupleType:
            self += Link(item[0],item[1])
        else:
            raise TypeError("Multplication operator for logs (adding links) can only be used with PATH or (PATH,LINKNAME)")
        return self
            
        
    def __getitem__(self,k):
        if k in self._sections:
            return self._sublogs[k]
        else:
            raise KeyError("Section %s does not exist" % (k,))

    def __setitem__(self,k,value):
        if isinstance(value,ExperimentLog):
            self._sublogs[k] = value
        else:
            raise TypeError("Assignment value must be a ExperimentLog")


    def __str__(self):
        return self.ascii()

    def ascii(self):
        """
        :returns: An ascii representation of the experiment log.
        :rtype: string
        """
        joinfunc = lambda x: textwrap.fill(x,80) if type(x) == types.StringType else x.ascii()

        s = "\n%s\n" % (self._name.upper(),)
        s +=  " ".join([joinfunc(elem) for elem in self._log]) + "\n"

        for k in self._sections:
            s += 80*"-" + "\n"
            s += self._sublogs[k].ascii()
            

        return s

    def html(self, title=True):
        """
        :returns: An html representation of the experiment log.
        :rtype: string
        """
        joinfunc = lambda x: x if type(x) == types.StringType else x.html()
        s = "<table border=\"0\" width=\"80%\" cellspacing=\"5\">"
        if len(self._sublogs) > 0:
            if title:
                s += "<tr><td align=\"center\" colspan=\"2\"><b>%s</b></td></tr>" % (self._name,)
            s += "<tr><td colspan=\"2\">" + "\n".join([joinfunc(elem) for elem in self._log]) + "</td></tr>"
        else:
            if title:
                s += "<tr><td align=\"center\"><b>%s</b></td></tr>" % (self._name,)
            s += "<tr><td>" + "\n".join([joinfunc(elem) for elem in self._log]) + "</td></tr>"

        for k in self._sections:
            s += "<tr><td valign=\"top\"><b>%s</b></td>" % (k,)
            s += "<td valign=\"top\">%s</td></tr>" % (self[k].html(title=False))
        s += "</table>"
        return s


    def __repr__(self):
        return self.__str__()
        
    def write(self,filename,format='html'):
        """
        Writes the experiment log to a file.

        :param filename: Name of the output file.
        :type filename: string
        :param format: Output format. Possible formats are: html, ascii
        :type format: string
        """
        f =  open(filename,'w')
        f.write(getattr(self,"_%s_HEADER" % (format.upper(),)) % (self._name,))
        f.write(self.__log__(format))
        f.write(getattr(self,"_%s_FOOTER" % (format.upper(),)) )
        f.close()

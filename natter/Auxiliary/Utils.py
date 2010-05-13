import pickle


def save(o,filename):
    tmp = filename.split('.')
    if tmp[-1] == 'pydat':
        f = open(filename,'w')
    else:
        f = open(filename + '.pydat','w')
        
    pickle.dump(o,f)
    f.close()

def testProtocol(value):
    s = "\n"
    s+= "++++++++++++++++++++++++ Test Error Protocol ++++++++++++++++++++++++\n"
    for (k,v) in value.items():
        s += str(k).upper()  + ": \n"
        s += str(v) + '\n'
        s += 10*'- - ' + '\n'
    s += "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    return s


HaveIpython=True
try:
    from IPython.Debugger import Tracer;  debug = Tracer()
except:
    HaveIpython=False
    def debug():
        pass
    pass

    
 

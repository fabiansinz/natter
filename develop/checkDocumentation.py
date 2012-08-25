import natter
import types



def checkDocumentation(mod,checked=[]):
    # print "checking " + mod.__name__
    # raw_input()
        
    children = [elem for elem in dir(mod) if not elem.startswith('_')]
    for submod in children:
        smod = getattr(mod,submod)
        if (smod is not None) and (hasattr(smod,'__package__') and smod.__package__ is not None and smod.__package__.startswith('natter') )\
               or (hasattr(smod,'__module__') and smod.__module__ is not None and smod.__module__.startswith('natter') ):

            
            name = smod.__name__
            if not name.startswith('natter'):
                if hasattr(smod,'__package__'):
                    name =  smod.__name__
                elif hasattr(smod,'__module__'):
                    name =  smod.__module__ + '.' + smod.__name__


            if not name in checked:
                if smod.__doc__ is None:
                    print  "   + [ ] "  + name
                checked.append(name)
                checked = checkDocumentation(smod,checked)
        
    return checked
    

if __name__=="__main__":
    checkDocumentation(natter)

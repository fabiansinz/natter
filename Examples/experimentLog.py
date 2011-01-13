from natter.Logging import ExperimentLog
from natter.Logging.LogTokens import *



if __name__=="__main__":

    p = ExperimentLog("Crucial Experiement")

    p += 'This experiment was really awesome. You can read about it on '
    p *= ('http://www.nature.com','Nature')

    p.addSection('Setup')
    p['Setup'] += "We had this super setup, which consisted of ..."

    p.addSection('Software')
    p['Software'] += "We used the natter version"
    p['Software'] /= Git('../')
    p['Software'] /= "We used Python version"
    p['Software'] /= PyInfo()
    
    p.addSection('Results')
    T = Table((1,2),('train','test'))
    T[1,'train'] = 1.0
    T[2,'test'] = 2.0
    p['Results'] /= T
    
    print p.ascii()

    p.write('test.html','html')
    

from natter.Logging import *
from natter.Logging import ExperimentLog

if __name__=="__main__":

    log = ExperimentLog("My Experiment Log")

    log += """In this experiment, we did a  lot of important stuff!"""

    log *= ("myfile.html","here")
    
    log.addSection("Data")
    log['Data'] += """We sampled ten datasets of 100.000 7x7 patches from the van Hateren
                      dataset. The data was preprocessed with data preprocessing protocol 1
                      and whitened with whitening protocol 1."""
    log.addSection("Experiments")
    log['Experiments'] += """For computing the tuning curves, we determined the response to
                            different orientations by computing the
                            response of the linear filter to a grating
                            directly in the Fourier domain (see
                            here). The test gratings for the different
                            orientations were chosen to correspond to
                            have optimal phase and clipped to points
                            in the DFT spectrum (therefore, the
                            frequency and orientation might differ a
                            little bit). We computed the linear
                            reponse of the filter to different
                            gratings with std one in that
                            way. Afterwards we rescaled the responses
                            with 6.63 (std of the data the filters
                            were trained on). After the rescaling we
                            put the linear responses to the radial
                            factorization non-linearity and plotted
                            the response curves."""

    log.addSection('Results')
    t = Table((1,2),('a','b'))
    t[1,'a'] = 1.23
    t[2,'a'] = "1.2612312"
    t[1,'b'] = Link("myfile.html","linkname")
    t[2,'b'] = 1.22

    log['Results'] += t

    print log.__log__('ascii')
    
    log.write('test.html','html')
    log.write('test.txt','ascii')

    

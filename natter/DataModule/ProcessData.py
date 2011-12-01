from __future__ import division
from copy import deepcopy as cp
from numpy import floor, zeros, squeeze

def getSizeSlice(sl,defStart,defStop):
    st = sl.start
    se = sl.stop
    sp = sl.step
    if not st:
        st = defStart
    if not se:
        se = defStop
    if not sp:
        sp =1
    return len(range(st,se,sp))

class ProcessData(object):
    """
    Special Data object namely process data.


    fields:
    dimObs  :  dimensionality of the observations
    
    """

    def __init__(self, *args , **kwargs):
        """
        Constructor for a general process data set.

        Arguments:
        :param *args: Variable length list of direct non-keyword sets.
                      The order is the following:
                      X : single array or list of arrays containing the observed values
                      at timepoints.
                      timePoints: discretization points
        :param **kwargs:

        ==================  =========================================================
        Key                  Description
        ==================  =========================================================
        'Values'            list of [n,Ti] arrays, where Ti is the discritization
                            length of the i-th trial n is the dimensionality of
                            the observations.
        ------------------  --------------------------------------------------------
        'Times'             List of arrays of length Ti containing the discretization
                            points for the values.
        ------------------  --------------------------------------------------------
        'Mode'              String. Can either be 'constant' or 'point'. It influences
                            how the values are interpreted. 'constant' means, that the
                            values are treated as constant between the time-points,
                            whereas 'point' means that only the value at that time
                            point is known. Default is 'point'.
        ==================  =========================================================
        
        """

        self.mode = 'point'
        self.nTrials    = 0
        
        if len(args)>0:
            if len(args)<2:
                raise AttributeError('If Values are given, then also time points have to be given.')
            
            values     = args[0]
            timePoints = args[1]
        elif len(kwargs.keys())>0:
            if not ('Values' in kwargs.keys() and 'Times' in kwargs.keys() ):
                raise AttributeError('Both \'Values\' and \'Times\' have to specified')
            values     = kwargs['Values']
            timePoints = kwargs['Times']
            if 'Mode' in kwargs.keys():
                self.mode =kwargs['Mode']
        else:
            raise ValueError('No Values given!')

        if isinstance(values,list):
            self.dimObs     = values[0].shape[0]
            self.values     = []
            self.timePoints = []
            for i,V in enumerate(values):
                tP = timePoints[i]
                self.addTrial(V,tP)
        else:
            self.dimObs = values.shape[0]
            self.values     = []
            self.timePoints = []
            self.addTrial(values,timePoints)

    def addTrial(self,values,timePoints):
        """
        Add a trial consisting of observations and time points to the dataset.
        
        Arguments:
        :param values: n x Ti array containing the observations
        :type values:  numpy.ndarray

        :param timePoints: Array containing the discretization points.
        :type timePoints: numpy.ndarray


        """
        if len(values.shape)==1:
            d = values.shape[0]
            values =values.reshape(d,1)
        if values.shape[0] != self.dimObs:
            raise AttributeError('Dimensionality of the observations does not match the previous ones!')
        if values.shape[1]!=len(timePoints):
            raise AttributeError('Length of time discretization  does not match the length of the values.')
        self.timePoints.append(timePoints)
        self.values.append(values)
        self.nTrials +=1

    def __getitem__(self, item):
        if len(item)==3:
            obs,timeP,trials =  item
            if isinstance(trials, slice):
                values = []
                if trials.start is None:
                    start = 0
                if trials.stop is None:
                    stop = self.nTrials
                if trials.step is None:
                    step =1
                szTr = getSizeSlice(trials,0,self.nTrials)
                if isinstance(timeP,slice):
                    szTime= getSizeSlice
                else:
                    szTime=1
                if isinstance(obs,slice):
                    szObs = getSizeSlice(obs,0,self.dimObs)
                else:
                    szObs = 1
                values = zeros((szObs,szTime,szTr))
                for k in range(start,stop,step):
                    values[obs,timeP,k]=self.values[k][obs,timeP]
                return squeeze(values)
            elif isinstance(trials,int):
                return self.values[trials][obs,timeP]
        elif len(item)==1:              # extract only the given trials
            return  ProcessData({'Values':self.values[item] ,
                                 'Times': self.timePoints[item]})
    
    
    


    
    
    
                
        

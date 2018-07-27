"""
This is a Python 3 implementation of the Predictive Sequence Learning algorithm, 
using a channel coded data representation. 
"""

import numpy as np
from chanpy import Cos2ChannelBasis

class ChanPsl(object):
    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, memory=None):
        self.memory = memory # Initated when first sample is provided
        self.__basis__ = channelBasis or Cos2ChannelBasis()
        if not channelBasis: 
            self.__basis__.setParameters(nChannels, minValue, maxValue)
        
    def encodeSample(self,s):
        v = self.__basis__.encode(s)
        if self.memory is None:
            self.memory = np.zeros([v.size,v.size])
        else: 
            assert v.size**2 == self.memory.size
        return v

    def trainSample(self,history,target):
        v = self.encodeSample(history[-1])
        t = self.encodeSample(target)
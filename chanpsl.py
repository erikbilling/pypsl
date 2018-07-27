"""
This is a Python 3 implementation of the Predictive Sequence Learning algorithm, 
using a channel coded data representation. 
"""

import numpy as np
from chanpy import Cos2ChannelBasis, ChannelVector

class ChanPsl(object):
    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, memory=None):
        self.memory = memory # Initated when first sample is provided
        self.__basis__ = channelBasis or Cos2ChannelBasis()
        if not channelBasis: 
            self.__basis__.setParameters(nChannels, minValue, maxValue)
        
    def encode(self,s):
        return ChannelVector(self.__basis__) if s is None else self.__basis__.encode(s)

    def trainSample(self,history,target):
        v = self.encode(history[-1])
        t = self.encode(target)
        if self.memory is None:
            self.memory = np.zeros([v.size,v.size])
        else: 
            assert v.size**2 == self.memory.size
        self.memory += v.transpose()*t/1.125

    def predict(self,history):
        return self.predictSample(history[-1] if history else None)

    def predictSample(self,value):
        res = self.predictVector(self.encode(value))
        return res.decode().ravel()[0]

    def predictVector(self,v):
        res = ChannelVector(self.__basis__)
        res[:] = np.dot(v,self.memory + np.random.rand(*self.memory.shape)*10)
        return res
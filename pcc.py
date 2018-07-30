"""
This is a Python 3 implementation of the Predictive Sequence Learning algorithm, 
using a channel coded data representation. 
"""

import numpy as np
from chanpy import Cos2ChannelBasis, ChannelVector

class Pcc(object):
    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, memory=None):
        self.memory = memory # Initated when first sample is provided
        self.__basis__ = channelBasis or Cos2ChannelBasis()
        if not channelBasis: 
            self.__basis__.setParameters(nChannels, minValue, maxValue)
        
        self.memory = np.zeros([self.__basis__.getNrChannels(),self.__basis__.getNrChannels()])
        self.memory2 = np.zeros([self.__basis__.getNrChannels(),self.__basis__.getNrChannels()])

    @property
    def minValue(self):
        return self.__basis__.getMinV()

    @property
    def maxValue(self):
        return self.__basis__.getMaxV()

    def encode(self,s):
        if s is None:
            return ChannelVector(self.__basis__)
        elif isinstance(s,InputTrace):
            return s.__cv__
        elif isinstance(s,ChannelVector):
            return s
        elif isinstance(s,list):
            e = ChannelVector(self.__basis__)
            for i,v in enumerate(s): 
                e += self.encode(v) / (len(s)-i)
            return e
        else:
            return self.__basis__.encode(s)

    def decode(self,v):
        cv = ChannelVector(self.__basis__)
        cv[:] = v
        return cv.decode().ravel()[0]

    def encodeInput(self,i):
        if isinstance(i,InputTrace):
            return i.__last__
        elif isinstance(i,list):
            return self.encode(i[-1])
        else:
            return self.encode(i)

    def encodeContext(self,i):
        return i.__trace__ if isinstance(i,InputTrace) else self.encode(i[-50:-1])

    def trainSample(self,history,target):
        v = self.encodeInput(history)
        context = self.encodeContext(history)
        t = self.encode(target)
        p = self.predictVector(v,context)
        err = t-p
        #err[err<0] = 0
        err2 = p-t
        err2[err2<0] = 0
        self.memory += v.transpose()*(err/1.125)/100.
        self.memory2 += context.transpose()*(err/1.125/100.)
        return err

    def predict(self,val,decode=True):
        res = self.predictVector(self.encodeInput(val),self.encodeContext(val))
        return res.decode().ravel()[0] if decode else res

    def predictSample(self,val,context=None,decode=True):
        res = self.predictVector(self.encodeInput(val),context or self.encodeContext(val))
        return res.decode().ravel()[0] if decode else res

    def predictVector(self,v,context=None):
        #print(v.shape,self.memory.shape,context.shape,self.memory2.shape)
        res = ChannelVector(self.__basis__)
        res[:] = np.dot(v,self.memory)
        if context is not None:
            res[:] += np.dot(context,self.memory2)
        return res

class InputTrace(object):
    def __init__(self,pcc,retention=0.5):
        self.__pcc__ = pcc
        self.__trace__ = ChannelVector(pcc.__basis__)
        self.__last__ = ChannelVector(pcc.__basis__)
        self.retention = retention

    def addSample(self,val):
        self.__trace__ *= self.retention
        self.__trace__ += self.__last__
        self.__last__ = self.__pcc__.encode(val)
        return self 

    def reset(self):
        self.__init__(self.__pcc__,self.retention)           


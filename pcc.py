"""
This is a Python 3 implementation of Predictive Channel Coding, 
based on chanpy for channel coded data representation. 
"""

import numpy as np
from chanpy import Cos2ChannelBasis, ChannelVector
from collections import Collection
import itertools
from random import random

class Pcc(object):
    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, memory=None, learningRate=0.01, noise=None):
        self.memory = memory # Initated when first sample is provided
        self.learningRate = learningRate
        self.noise = noise
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
        elif isinstance(s,ChannelVector):
            return s
        elif isinstance(s,Collection):
            e = ChannelVector(self.__basis__)
            for i,v in enumerate(s): 
                e += self.encode(v) / (len(s)-i)
            return e
        else:
            s += random()*self.noise-self.noise/2. if self.noise else 0
            cv = self.__basis__.encode(s)
            #if self.noise:
            #    cv[:] += np.random.rand(cv.size) * self.noise
            return cv

    def decode(self,v):
        cv = ChannelVector(self.__basis__)
        cv[:] = v
        return cv.decode().ravel()[0]

    def encodeInput(self,i):
        if isinstance(i,InputTrace):
            return i.__last__
        elif isinstance(i,Collection):
            return self.encode(i[-1])
        else:
            return self.encode(i)

    def encodeContext(self,i):
        return i.__trace__ if isinstance(i,InputTrace) else self.encode(i[-50:-1])

    def train(self,val,target):
        v = self.encodeInput(val)
        context = self.encodeContext(val)
        t = self.encode(target)
        p = self.predictVector(v,context)
        err = t-p
        self.memory += v.transpose()*(err/1.125*self.learningRate)
        self.memory2 += context.transpose()*(err/1.125*self.learningRate)
        return err

    def predict(self,val,decode=True):
        res = self.predictVector(self.encodeInput(val),self.encodeContext(val))
        res[res<0] = 0
        return res.decode().ravel()[0] if decode else res

    def predictVector(self,v,context=None):
        #print(v.shape,self.memory.shape,context.shape,self.memory2.shape)
        res = ChannelVector(self.__basis__)
        res[:] = np.dot(v,self.memory)
        if context is not None:
            res[:] += np.dot(context,self.memory2)
        return res

    def newTrace(self,values,retention=0.8):
        return InputTrace(self,retention,values)

    def trace(self,data,retention=0.5):
        t = InputTrace(self,retention)
        for v in data:
            yield (t,v)
            t.addSample(v)

    def gen(self,data,length=100,retention=0.8):
        t = InputTrace(self,retention,data)
        for v in data:
            yield v
        for i in itertools.count():
            if i + len(data) >= length: break
            p = self.predict(t)
            yield p
            t.addSample(p)

class InputTrace(object):
    def __init__(self,pcc,retention=0.5,val=None,decodePrediction=True):
        self.length = 0
        self.__pcc__ = pcc
        self.__trace__ = ChannelVector(pcc.__basis__)
        self.__last__ = ChannelVector(pcc.__basis__)
        self.retention = retention
        self.__initVal__ = val
        self.decodePrediction = decodePrediction
        self.addSample(val)

    def addSample(self,val):
        if isinstance(val,Collection): 
            for v in val: self.addSample(v)
        elif val is not None:
            self.__trace__ *= self.retention
            self.__trace__ += self.__last__
            self.__last__ = self.__pcc__.encode(val)
            self.length += 1
            return self

    def reset(self):
        self.__init__(self.__pcc__,self.retention,val=self.__initVal__)          

    def predict(self):
        return self.__pcc__.predict(self,self.decodePrediction)

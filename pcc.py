"""
This is a Python 3 implementation of Predictive Channel Coding, 
based on chanpy for channel coded data representation. 
"""

import numpy as np
from chanpy import Cos2ChannelBasis, ChannelVector
from collections import Collection
import itertools
from abc import ABC, abstractmethod

class Pcc:
    def __init__(self, nChannels=11, minValue=0., maxValue=1., encoder=None, memory=None, learningRate=0.01):
        if isinstance(nChannels, AbstractChannelEncoder): encoder = nChannels
        self.encoder = encoder or UniformChannelEncoder(nChannels, minValue, maxValue)
        self.memory = memory # Initated when first sample is provided
        self.learningRate = learningRate
        
        self.memory = np.zeros([self.encoder.channelCount,self.encoder.channelCount])
        self.memory2 = np.zeros([self.encoder.channelCount,self.encoder.channelCount])

    def train(self,val,target):
        v = self.encoder.encodeInput(val)
        context = self.encoder.encodeContext(val)
        t = self.encoder.encode(target)
        p = self.predictVector(v,context)
        err = t-p
        self.memory += v.transpose()*(err/1.125*self.learningRate)
        self.memory2 += context.transpose()*(err/1.125*self.learningRate)
        return err

    def predict(self,val,decode=True):
        res = self.predictVector(self.encoder.encodeInput(val),self.encoder.encodeContext(val))
        res[res<0] = 0
        return res.decode().ravel()[0] if decode else res

    def predictVector(self,v,context=None):
        res = self.encoder.newChannelVector()
        res[:] = np.dot(v,self.memory)
        if context is not None:
            res[:] += np.dot(context,self.memory2)
        return res

    def trace(self,data,retention=0.5):
        t = InputTrace(self.encoder,retention)
        for v in data:
            yield (t,v)
            t.addSample(v)

    def gen(self,data,length=None,retention=0.8,includeSourceData=False):
        if length is None: length=len(data)
        t = InputTrace(self.encoder,retention)

        for v in data:
            if includeSourceData: yield t,v
            t.addSample(v)
            if t.length >= length and includeSourceData: break

        while t.length < length + (len(data) if not includeSourceData else 0):
            p = self.predict(t)
            yield t,p
            t.addSample(p)

class InputTrace(object):
    def __init__(self,encoder,retention=0.8,val=None,decodePrediction=True):
        self.length = 0
        self.encoder = encoder
        self.__trace__ = encoder.newChannelVector()
        self.__last__ = encoder.newChannelVector()
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
            self.__last__ = self.encoder.encode(val)
            self.length += 1
            return self

    def reset(self):
        self.__init__(self.encoder,self.retention,val=self.__initVal__)

class AbstractChannelEncoder(ABC):

    def __init__(self,channelBasis):
        self.__basis__ = channelBasis

    @abstractmethod
    def encode(self,s):
        pass

    @abstractmethod
    def decode(self,v):
        pass

    @abstractmethod
    def encodeInput(self,i):
        pass

    @abstractmethod
    def encodeContext(self,i):
        pass

    def newChannelVector(self):
        return ChannelVector(self.__basis__)

class UniformChannelEncoder(AbstractChannelEncoder):
    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, noise=None):
        super().__init__(channelBasis or Cos2ChannelBasis())
        if not channelBasis: 
            self.__basis__.setParameters(nChannels, minValue, maxValue)
        self.noise = noise

    @property
    def minValue(self):
        return self.__basis__.getMinV()

    @property
    def maxValue(self):
        return self.__basis__.getMaxV()

    @property
    def channelCount(self):
        return self.__basis__.getNrChannels()

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
            if callable(self.noise):
                s += self.noise()
            elif self.noise:
                s += np.random.randn()*self.noise-self.noise/2.
            cv = self.__basis__.encode(s)
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
        return i.__trace__ if isinstance(i,InputTrace) else None
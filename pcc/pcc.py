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
        return self.encoder.decode(res) if decode else res

    def predictVector(self,v,context=None):
        res = self.encoder.newChannelVector()
        res[:] = np.dot(v,self.memory)
        if context is not None:
            res[:] += np.dot(context,self.memory2)
        return res

    def trace(self,data,retention=0.8,buffer=10):
        t = InputTrace(self.encoder,retention)
        if isinstance(self.encoder,AbstractDiffEncoder):
            b = buffer if isinstance(buffer,MeanBuffer) else MeanBuffer(buffer)
            for v in data:
                dv = v-b.mean()
                yield (t,v,dv)
                t.addSample(dv)
                b.put(v)
        else:
            for v in data:
                yield (t,v)
                t.addSample(v)

    def gen(self,data,length=None,retention=0.8,buffer=10,includeSourceData=False):
        if length is None: length=len(data)
        t = InputTrace(self.encoder,retention)

        if isinstance(self.encoder,AbstractDiffEncoder):
            b = buffer if isinstance(buffer,MeanBuffer) else MeanBuffer(buffer)
            for v in data:
                dv = v-b.mean()
                if includeSourceData: yield t,v,dv
                t.addSample(dv)
                b.put(v)
                if t.length >= length and includeSourceData: 
                    break
                
            while t.length < length + (len(data) if not includeSourceData else 0):
                dv = self.predict(t)
                v = b.mean()+dv
                yield t,v,dv
                t.addSample(dv)
                b.put(v)
        else:
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
        if isinstance(val,(list,tuple)): 
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

    def reset(self):
        pass

    @property
    def minValue(self):
        return self.__basis__.getMinV()

    @property
    def maxValue(self):
        return self.__basis__.getMaxV()

    @property
    def channelCount(self):
        return self.__basis__.getNrChannels()

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

class AbstractDiffEncoder(AbstractChannelEncoder):
    pass

class UniformChannelEncoder(AbstractChannelEncoder):
    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, noise=None):
        super().__init__(channelBasis or Cos2ChannelBasis())
        if not channelBasis: 
            self.__basis__.setParameters(nChannels, minValue, maxValue)
        self.noise = noise

    def encode(self,s):
        if s is None:
            return ChannelVector(self.__basis__)
        elif isinstance(s,ChannelVector):
            return s
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
        else:
            return self.encode(i)

    def encodeContext(self,i):
        return i.__trace__ if isinstance(i,InputTrace) else None


class LogDiffEncoder(AbstractDiffEncoder):
    def __init__(self, nChannels=11, minValue=-1, maxValue=1., noise=None):
        super().__init__(Cos2ChannelBasis())
        self.__trueMax__ = (maxValue-minValue)
        self.__trueMin__ = -self.__trueMax__
        self.__basis__.setParameters(nChannels, self.reshape(self.__trueMin__), self.reshape(self.__trueMax__))
        self.noise = noise

    @property
    def minValue(self):
        return self.__trueMin__

    @property
    def maxValue(self):
        return self.__trueMax__

    def reshape(self,value):
        return np.log(1. + np.abs(value)) * np.sign(value)

    def restore(self,value):
        return (np.exp(np.abs(value)) - 1.) * np.sign(value)

    def encode(self,s):
        if s is None:
            return self.newChannelVector()
        elif isinstance(s,ChannelVector):
            return s
        else:
            if callable(self.noise):
                s += self.noise()
            elif self.noise:
                s += np.random.randn()*self.noise-self.noise/2.
            cv = self.__basis__.encode(self.reshape(s))
            return cv

    def decode(self,v):
        cv = ChannelVector(self.__basis__)
        cv[:] = v
        return self.restore(cv.decode().ravel()[0])

    def encodeInput(self,i):
        if isinstance(i,InputTrace):
            return i.__last__
        else:
            return self.encode(i)

    def encodeContext(self,i):
        return i.__trace__ if isinstance(i,InputTrace) else None

class MeanBuffer:
    def __init__(self,size):
        self.buffer = np.zeros(size)
        self.i = -1
        self.n = 0

    @property
    def size(self):
        return self.buffer.size

    def mean(self):
        if self.n == 0:
            return 0
        elif self.n < self.buffer.size:
            return self.buffer[:self.n].mean()
        else:
            return self.buffer.mean()

    def put(self,v):
        self.i = (self.i+1)%self.buffer.size
        self.n += 1
        self.buffer[self.i] = v


def integrate(data,initv=None,buffer=10):
    b = buffer if isinstance(buffer,MeanBuffer) else MeanBuffer(buffer)
    if initv is not None: b.put(initv)
    for dv in data:
        v = b.mean()+dv
        yield v
        b.put(v)
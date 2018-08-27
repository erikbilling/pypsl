"""
This is an extension of pypsl that uses channel representations. 
"""

import numpy as np
from functools import reduce
from pypsl import Psl, Library, Hypothesis, LengthSelector, AbstractSelector
from chanpy import Cos2ChannelBasis, ChannelVector

# class ChanSelector(AbstractSelector):

#     def select(self, hypotheses, default=None):
        
CREATION_THRESHOLD = 0.01

class ChanHypothesis(Hypothesis):

    def confidence(self):
        pass

class ChanPsl(Psl):

    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, library=None, selector=None):
            super().__init__(library, selector)
            self.__basis__ = channelBasis or Cos2ChannelBasis()
            if not channelBasis:
                self.__basis__.setParameters(nChannels, minValue, maxValue)

    def add(self,lhs,rhs,baseLhs=()):
        lhs = lhs.ravel()
        rhs = rhs.ravel()
        for li in np.flatnonzero(lhs):
            lhsKey = (li,) + baseLhs
            for ri in np.flatnonzero(rhs):
                self.addOne(lhsKey,lhs[li],ri,rhs[ri])

    def addOne(self,lhsIndex,lhsValue,rhsIndex,rhsValue):
        if not isinstance(lhsIndex,tuple): lhsIndex = (lhsIndex,)
        self.library.add(lhsIndex,rhsIndex,lhsValue,miss(lhsValue,rhsValue))

    def train(self, s, startIndex=1, stopIndex=0):
        """Trains PSL on the sequence s, covering the range startIndex to stopIndex"""
        
        for i in range(startIndex,stopIndex or len(s)):
            target = s[i].ravel()
            match = list(self.match(s[:i]))

            prediction = self.predict(match=longest(match),decode=False)
            error = (s[i]-prediction).ravel()
            #correct = [h for h in match if target[h.rhs] <= h.confidence]
            
            # Adjusting weights of existing hypotheses
            for h in match:
                lhsValue = impact(h.lhs,s,i)
                rhsValue = target[h.rhs]
                h.reward(lhsValue)
                h.punish(miss(lhsValue,rhsValue))

            # Creating new hypotheses
            for rhsIndex in np.flatnonzero(error>=CREATION_THRESHOLD):
                hi = [h for h in match if h.rhs == rhsIndex]
                existingRoots = set()
                #maxlen = reduce(lambda length,h: len(h) if len(h)>length else length, hi, 0)                   
                
                # Extending existing hypotheses
                for h in hi:
                    existingRoots.add(h.lhs[-1])
                    for lhsIndex in np.flatnonzero(s[i-1-len(h)]):
                        lhs = (lhsIndex,) + h.lhs
                        lhsValue = impact(lhs,s,i)
                        self.addOne(lhs,lhsValue,rhsIndex,target[rhsIndex])

                # Adding new root hypotheses
                for lhsIndex in np.flatnonzero(s[i-1]):
                    if lhsIndex in existingRoots: continue
                    self.addOne(lhsIndex,s[i-1].ravel()[lhsIndex],rhsIndex,target[rhsIndex])


    def match(self,s):
        """Returns an iterator over all hypotheses matching specified sequence s"""
        nonzeros = []
        for hlen in range(1,len(s)+1):
            v = s[-hlen]
            nonzeros.insert(0,np.flatnonzero(v))
            matchingHypotheses = 0
            for combo in combine(nonzeros):
                hs = self.library.match(combo)
                for h in hs:
                    matchingHypotheses += 1
                    yield h
            if not matchingHypotheses: break

    def predict(self,s=None,match=None,decode=True):
        if match is None: match = longest(self.match(s))
        cv = ChannelVector(self.__basis__)
        cvflat = cv.ravel()
        for h in match:
            cvflat[h.rhs] += h.confidence
        return cv.decode().ravel()[0] if decode else cv

    def encode(self,v):
        if isinstance(v,(list,tuple)):
            return [self.encode(x) for x in v]
        else:
            return self.__basis__.encode(v)

# Helper functions

def combine(vlist,listindex=0,combo=None):
        """Given a list of integer arrays vlist, returns an iterator over all combinations of values, one from each array."""
        if combo is None: combo = np.empty(len(vlist), dtype=int)
        if listindex < len(vlist) -1:
            for v in vlist[listindex]:
                combo[listindex]=v
                for cmb in combine(vlist,listindex+1,combo):
                    yield tuple(cmb)
        else:
            for v in vlist[listindex]:
                combo[listindex]=v
                yield tuple(combo)

def impact(lhs,s,i):
    n = len(lhs)
    lhsValue = 1.
    for lhsp,lhsi in enumerate(lhs):
        lhsValue *= s[i-n+lhsp].ravel()[lhsi]
    return lhsValue

def longest(match):
    """Returns the longest hypotheses from specified match"""
    lng = {}
    for h in match:
        parent = h.lhs[1:]+(h.rhs,)
        if parent in lng:
            del lng[parent]
        lng[h.lhs + (h.rhs,)] = h
    return lng.values()

def miss(lhsStrength,rhsStrength):
    return lhsStrength/rhsStrength-lhsStrength
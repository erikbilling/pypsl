"""
This is an extension of pypsl that uses channel representations. 
"""

import numpy as np
from pypsl import Psl, Library, Hypothesis, LengthSelector, AbstractSelector
from chanpy import Cos2ChannelBasis, ChannelVector

# class ChanSelector(AbstractSelector):

#     def select(self, hypotheses, default=None):
        

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
        self.library.add(lhsIndex,rhsIndex,lhsValue,miss(lhsValue,rhsValue))

    def train(self, s, startIndex=1, stopIndex=0):
        """Trains PSL on the sequence s, covering the range startIndex to stopIndex"""
        lenSelector = LengthSelector()
        s = [self.__basis__.encode(v) for v in s] # Encodes all values to channel vecotrs
        for i in range(startIndex,stopIndex or len(s)):
            target = s[i]
            match = list(self.match(s[:i]))
            if match:
                prediction = self.predict(match,decode=False)
                error = target-prediction
                correct = [h for h in match if target[h.rhs] <= h.confidence]
                for i in np.flatnonzero(error):
                    hi = [h for h in match if h.rhs == i]
                    longest = 0
                    for h in hi:
                        h.reward()
                        h.punish(miss(1.,target[i]))
                        longest = len(h) if len(h) > longest else longest


            
            correct = filter(lambda h: h.rhs == target, match)
            #incorrect = filter(lambda h: h.rhs != s[i], match)
            selected = self.library.selector.select(match)
            predictionCorrect = selected and selected.rhs == target
            if not predictionCorrect:
                if selected: selected.punish()
                selectedCorrect = lenSelector.select(correct)
                if not selectedCorrect:
                    self.library.add(s[i-1],target)
                elif len(selectedCorrect) <= len(selected):
                    self.library.add(s[i-1-len(selectedCorrect):i],target)
            for h in correct: 
                h.reward()

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

    def predict(self,s,decode=True):
        hypotheses = self.match(s)
        cv = ChannelVector(self.__basis__)
        cvflat = cv.ravel()
        for h in hypotheses:
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

def miss(rhsStrength,lhsStrength):
    return rhsStrength/lhsStrength-rhsStrength
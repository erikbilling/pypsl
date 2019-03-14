"""
This is an extension of pypsl that uses channel representations. 
"""

import numpy as np
from functools import reduce
from pypsl import Psl, Library, AbstractHypothesis, DefaultSelector
from chanpy import Cos2ChannelBasis, ChannelVector

# class ChanSelector(AbstractSelector):

#     def select(self, hypotheses, default=None):
        
CREATION_THRESHOLD = 0.01

class ChanHypothesis(AbstractHypothesis):

    def __init__(self,lhs,rhs,strength,coverage=1):
        super().__init__(lhs,rhs)
        self.strength = strength
        self.coverage = coverage

    def __repr__(self):
            return '{0}=>{1}({2:.2f})'.format(repr(self.lhs),repr(self.rhs),self.strength)

    @property
    def confidence(self):
        return self.strength/self.coverage

class ChanLibrary(Library):

    def __init__(self,items=None,selector=DefaultSelector(),hypothesisClass=ChanHypothesis):
        super().__init__(items,selector,hypothesisClass)

class ChanPsl(Psl):

    def __init__(self, nChannels=11, minValue=0., maxValue=1., channelBasis=None, library=None, selector=None):
            super().__init__(library or ChanLibrary(), selector)
            self.__basis__ = channelBasis or Cos2ChannelBasis()
            if not channelBasis:
                self.__basis__.setParameters(nChannels, minValue, maxValue)

    def add(self,lhs,rhs,baseLhs=()):
        lhs = lhs.ravel()
        rhs = rhs.ravel()
        for li in np.flatnonzero(lhs):
            lhsKey = (li,) + baseLhs
            for ri in np.flatnonzero(rhs):
                self.addOne(lhsKey,ri,rhs[ri])

    def addOne(self,lhsIndex,rhsIndex,strength):
        if not isinstance(lhsIndex,tuple): lhsIndex = (lhsIndex,)
        self.library.add(lhsIndex,rhsIndex,strength,1)

    def train(self, s, startIndex=1, stopIndex=0):
        """Trains PSL on the sequence s, covering the range startIndex to stopIndex"""
        
        for i in range(startIndex,stopIndex or len(s)):
            target = s[i].ravel()
            match = list(self.match(s[:i]))
            #longMatch = longest(match)
            prediction = self.predict(s[:i],match=match,decode=False)
            error = (s[i]-prediction).ravel()
            #correct = [h for h in match if target[h.rhs] <= h.confidence]
            
            # Adjusting weights of existing hypotheses
            for h in match:
                #h.coverage += 1
                if error[h.rhs] > 0:
                    h.strength += error[h.rhs] * support(h.lhs,s,i)

            # Creating new hypotheses
            for rhsIndex in np.flatnonzero(error):
                hi = [h for h in match if h.rhs == rhsIndex]
                existingRoots = set()
                #maxlen = reduce(lambda length,h: len(h) if len(h)>length else length, hi, 0)                   
                
                # Extending existing hypotheses
                for h in hi:
                    existingRoots.add(h.lhs[-1])
                    if error[h.rhs] > 0: continue
                    for lhsIndex in np.flatnonzero(s[i-1-len(h)]):
                        lhs = (lhsIndex,) + h.lhs
                        if lhs in self.library.__lib__: continue
                        self.addOne(lhs,rhsIndex,error[rhsIndex])

                # Adding new root hypotheses
                for lhsIndex in np.flatnonzero(s[i-1]):
                    if lhsIndex in existingRoots: continue
                    self.addOne(lhsIndex,rhsIndex,error[rhsIndex])


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
            cvflat[h.rhs] += h.confidence*support(h.lhs,s)
        return cv.decode().ravel()[0] if decode else cv

    def encode(self,v):
        if isinstance(v,(list,tuple)):
            return [self.encode(x) for x in v]
        else:
            return self.__basis__.encode(v)

    def decode(self,v):
        if isinstance(v,np.ndarray):
            cv = ChannelVector(self.__basis__)
            cv[:] = v
            return cv.decode().ravel()[0]
        else:
            return [self.decode(i) for i in v]
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

def support(lhs,s,i=0):
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
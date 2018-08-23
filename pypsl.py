"""
This is a Python 3 implementation of Predictive Sequence Learning (PSL)
"""

from abc import ABC, abstractmethod

class Psl:

    def __init__(self,library=None,selector=None):
        self.library = library if library is not None else Library(selector if selector is not None else DefaultSelector())

    def predict(self,s):
        """Returns the next element in the specified sequence s"""
        h = self.select(s)
        return h.rhs if h else None

    def train(self,s, startIndex=1, stopIndex=-1):
        """Trains PSL on the sequence s, covering the range startIndex to stopIndex"""
        pass

    def match(self,s):
        """Returns an iterator over all hypotheses matching specified sequence s"""
        hlen = 1
        while True:
            hs = self.library.match(s[-hlen:])
            if not hs: break
            for h in hs:
                yield h
            hlen += 1

    def select(self,s,default=None):
        return self.library.selector.select(self.library.match(s),default)

class AbstractSelector:
    """Selects the best hypotheses from a set of matching hypotheses"""

    @abstractmethod
    def select(self,hypotheses,default=None):
        pass

class DefaultSelector(AbstractSelector):

    def select(self,hypotheses,default=None):
        maxConf = -1
        bestHypothesis = default
        for h in hypotheses.values():
            if h.confidence > maxConf or (h.confidence == maxConf and h.hits > bestHypothesis.hits):
                maxConf = h.confidence
                bestHypothesis = h
        return bestHypothesis

class Library:
    def __init__(self,items=None,selector=DefaultSelector()):
        self.__lib__ = dict()
        self.selector = selector
        if items: 
            for key, value in (items.items() if isinstance(items,dict) else items):
                self.add(key,value)

    def __len__(self):
        count = 0
        for s in self.__lib__.values():
            count+=len(s)
        return count

    def __repr__(self):
        return repr(self.__lib__)

    def __getitem__(self,key):
        h = self.selector.select(self.match(key))
        if h is None: 
            raise KeyError(key)
        return h.rhs

    def get(self,key,default=None):
        h = self.selector.select(self.match(key),default)
        return h.rhs if isinstance(h,Hypothesis) else h

    def match(self,key,default=()):
        return self.__lib__.get(key,default)

    def add(self,key,value,hits=1,misses=0):
        s = self.__lib__.get(key)
        if s:
            h = s.get(value)
            if h:
                h.reward(hits)
                h.punish(misses)
            else:
                s[value] = Hypothesis(key,value,hits,misses)
        else:
            self.__lib__[key] = {value: Hypothesis(key,value,hits,misses)}
            

class Hypothesis:
    """Represents an hypothesis (a,b,c => d), with specified target and confidence"""

    def __init__(self,lhs,rhs,hits=1,misses=0):
        self.lhs = lhs
        self.rhs = rhs
        self.hits = hits
        self.misses = misses
        self.__hashCode__ = hash((lhs,rhs))

    def __hash__(self):
        return self.__hashCode__

    def __repr__(self):
        return '{0}=>{1}({2}/{3})'.format(repr(self.lhs),repr(self.rhs),self.hits,self.misses)

    def reward(self,hits=1):
        self.hits+=hits

    def punish(self,misses=1):
        self.misses+=misses

    @property
    def confidence(self):
        return len(self.lhs) * self.hits/(self.hits+self.misses)


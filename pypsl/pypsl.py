"""
This is a Python 3 implementation of Predictive Sequence Learning (PSL)
"""

from abc import ABC, abstractmethod
from collections import Hashable, Collection

class Psl:
    """Psl is the main class for the Predictive Sequence Learning (PSL) implementation. It provides methods for sequence learning and prediction."""

    def __init__(self,library=None,selector=None):
        self.library = library if library is not None else Library(selector = selector if selector is not None else DefaultSelector())

    def predict(self,s):
        """Returns the next element in the specified sequence s"""
        h = self.select(s)
        return h.rhs if h else None

    def formatSequence(self,s):
        if isinstance(s,str): 
            return tuple(s)
        return s
    
    def train(self, s, startIndex=1, stopIndex=0):
        """Trains PSL on the sequence s, covering the range startIndex to stopIndex"""
        lenSelector = LengthSelector()
        s = self.formatSequence(s)
        for i in range(startIndex,stopIndex or len(s)):
            match = list(self.match(s[:i]))
            target = s[i]
            correct = filter(lambda h: h.rhs == target, match)
            #incorrect = filter(lambda h: h.rhs != s[i], match)
            selected = self.library.selector.select(match)
            predictionCorrect = selected and selected.rhs == target
            if not predictionCorrect:
                if selected: selected.punish()
                selectedCorrect = lenSelector.select(correct)
                if not selectedCorrect:
                    self.library.add((s[i-1],),target)
                elif len(selectedCorrect) <= len(selected):
                    self.library.add(s[i-1-len(selectedCorrect):i],target)
            for h in correct: 
                h.reward()

    def match(self,s):
        """Returns an iterator over all hypotheses matching specified sequence s"""
        s = self.formatSequence(s)
        for hlen in range(1,len(s)+1):
            hs = self.library.match(s[-hlen:])
            if not hs: break
            for h in hs:
                yield h

    def select(self,s,default=None):
        return self.library.selector.select(self.match(s),default)

class AbstractHypothesis:

    def __init__(self,lhs,rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.__hashCode__ = hash((lhs,rhs))

    def __hash__(self):
        return self.__hashCode__

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.lhs)

    def __repr__(self):
        return '{0}=>{1}'.format(repr(self.lhs),repr(self.rhs))

    @property
    @abstractmethod
    def confidence(self):
        pass

class Hypothesis(AbstractHypothesis):
    """Represents an hypothesis (a,b,c => d), with acosiated confidence"""

    def __init__(self,lhs,rhs,hits=1,misses=0):
        super().__init__(lhs,rhs)
        self.hits = hits
        self.misses = misses

    def __repr__(self):
        return '{0}=>{1}({2:.2f}/{3:.2f})'.format(repr(self.lhs),repr(self.rhs),self.hits,self.misses)

    def reward(self,hits=1):
        self.hits+=hits

    def punish(self,misses=1):
        self.misses+=misses

    @property
    def confidence(self):
        return len(self.lhs) * self.hits/(self.hits+self.misses)

class AbstractSelector:
    """Selects the best hypotheses from a set of matching hypotheses"""

    @abstractmethod
    def select(self,hypotheses,default=None):
        pass

class DefaultSelector(AbstractSelector):
    """The detault selector selects the hypothesis with the highest confidence"""

    def select(self,hypotheses,default=None):
        maxConf = -1
        bestHypothesis = default
        for h in hypotheses:
            if h.confidence > maxConf or (h.confidence == maxConf and h.hits > bestHypothesis.hits):
                maxConf = h.confidence
                bestHypothesis = h
        return bestHypothesis

class LengthSelector(AbstractSelector):
    """Selects the longest hypothesis"""

    def select(self, hypotheses, default = None):
        maxLen = -1
        bestHypothesis = default
        for h in hypotheses:
            if len(h) > maxLen:
                maxLen = len(h)
                bestHypothesis = h
        return bestHypothesis

class Library:
    """Container class for all hypotheses."""

    def __init__(self,items=None,selector=DefaultSelector(),hypothesisClass = Hypothesis):
        self.__lib__ = dict()
        assert issubclass(hypothesisClass,AbstractHypothesis)
        self.__hypothesis_class__ = hypothesisClass
        self.selector = selector
        if items: 
            for key, value in (items.items() if isinstance(items,dict) else items):
                self.add(key,value)

    def __len__(self):
        count = 0
        for s in self.__lib__.values():
            count+=len(s)
        return count

    def __iter__(self):
        """Iterates over all hypotheses in library"""
        for s in self.__lib__.values():
            for h in s.values(): 
                yield h

    def __repr__(self):
        return repr(list(iter(self)))

    def __getitem__(self,key):
        h = self.selector.select(self.match(key))
        if h is None: 
            raise KeyError(key)
        return h.rhs

    def get(self,key,default=None):
        h = self.selector.select(self.match(key),default)
        return h.rhs if isinstance(h,Hypothesis) else h

    def match(self,key,default={}):
        key = self.__make_key__(key)
        return self.__lib__.get(key,default).values()

    def add(self,key,value,hits=1,misses=0):
        key = self.__make_key__(key)
        s = self.__lib__.get(key)
        if s:
            h = s.get(value)
            if h:
                h.reward(hits)
                h.punish(misses)
            else:
                s[value] = self.__hypothesis_class__(key,value,hits,misses)
        else:
            self.__lib__[key] = {value: self.__hypothesis_class__(key,value,hits,misses)}

    def __make_key__(self,key):
        if isinstance(key,list): 
            key = tuple(key) # Makes the sequence hashable
        elif not isinstance(key,Collection):
            key = (key,) # Wraps single values in tuple
        return key
            



import math, itertools
import numpy as np

def singen(length=100,start=0.,step=0.1):
    v = start
    for i in itertools.count():
        yield math.sin(v)
        v+=step
        if i >= length: 
            break

def trigen(length=100,start=0,step=1.,amplitude=1.,plateau=0):
    v = start
    count = itertools.count()
    for i in count:
        if i >= length: break
        yield v
        if abs(v) >= amplitude: 
            for j in range(plateau):
                if next(count) >= length: break
                yield v
            step = abs(step) * -1 * np.sign(step)
        v += step
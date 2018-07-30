import math

def singen(start=0,stop=100,step=0.1):
    i = start
    while stop is None or i<stop:
        yield math.sin(i)
        i+=step

def trigen(start=0,stop=100,step=1.,amplitude=1.,plateau=0):
    i = start
    v = 0
    yield v
    while stop is None or i<stop:
        v += step
        yield v
        if abs(v) >= amplitude: 
            for j in range(plateau): yield v
            step = step * -1
        i += 1

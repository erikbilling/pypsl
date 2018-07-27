import math

def singen(start=0,stop=100,step=0.1):
    i = start
    while stop is None or i<=stop:
        yield math.sin(i)
        i+=step


pypsl
=====

A Python implementation of Predictive Sequence Learning. 

Predictive Sequence Learning (**PSL**) is a machine learning algorithm for robot learning from demonstration and internal simulation of sensory-motor interactions. 

**NOTE:** This library currently only comprise an implementation of Predictive Channel Coding (**PCC**). While it has strong similarities to PSL it is a different algorithm. This may change in the future. 

Dependencies
------------

PCC depends on the following libraries: 
* Numpy
* [Chanpy library](https://github.com/micfe03/channel_representation)

Installation
------------

After installing dependencies, simply clone or download this package and place on your Python path. A pip package may be provided in the future. 

Usage
-----

**PCC** is intended for sequence prediction or reconstruction of continuous data and can be seen as a type of regression technique. Given a sequence of values, we want to predict the most probable next value of the sequence.

Let's start with a very basic example where *PCC* is used to predict a sinus curve:

~~~~python
    import numpy as np
    from test.inputdata import singen
    from pcc import Pcc

    pcc = Pcc(25,-1,1) # Creates an instance of Pcc with 25 channels, spanning over a single dimension from -1 and 1. 
    data = [v for v in singen(length=1000)] # And the sample data
    for trace,v in pcc.trace(data): # Creates an pcc.InputTrace which provides a channel code with decaying look-back. 
        pcc.train(trace,v) # Trains the model to associate the provided trace with the target value v.

    result = [pcc.predict(trace) for trace,v in pcc.trace(data)] # Compute 1-step predictions from data
    mse = np.square(np.array(result)-np.array(data)).mean() # Compare with the original data
    print('MSE: {0:.6f}'.format(mse))
~~~~


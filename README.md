pypsl
=====

A Python implementation of Predictive Sequence Learning. 

Predictive Sequence Learning (**PSL**) is a machine learning algorithm for robot learning from demonstration and internal simulation of sensory-motor interactions. 
See Billing et. al (2011) for a detailed algorithm description and demonstration. PSL has since then been extended in several ways, e.g. as demonstrated by Billing et al. (2012, 2016). 
A more comprehensive implementation of PSL in Java is available at https://bitbucket.org/interactionlab/psl, and a simple tutorial-implementation in Javascript is found at https://github.com/billingo/psl.js. 
You may also refer to http://cognitionreversed.com for examples and references. 

**pypsl** currently comprise a basic, descrete, version of the *PSL* algorithm (Billing et al. 2011). In addition, the package also comprise a more experimental implementation of Predictive Channel Coding (PCC). 

Dependencies
------------

**pypsl** is implemented for Python 3.X.
PSL has no additional dependencies. 

PCC depends on the following libraries: 
* Numpy
* [Chanpy library](https://github.com/micfe03/channel_representation)

Installation
------------

After installing dependencies, simply clone or download this package and place on your Python path. A pip package may be provided in the future. 

Usage
-----

Both **PSL** and **PCC** is intended for sequence prediction or reconstruction of continuous data and can be seen as a type of regression technique. Given a sequence of values, we want to predict the most probable next value of the sequence.

Let's start with a very basic example where *PSL* is used to predict a character sequence: 

~~~~python
from pypsl import Psl

s = 'abccabccabccabcc'
psl = Psl()
psl.train(s)
for i in range(100): 
    c = psl.predict(s)
    print(i,'Predicted:',c)
    s += c
print('Generated sequence: ',s)
~~~~

Let's continue with an example where *PCC* is used to predict a sinus curve:

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

This example is testing the model directly on the training data, so we should expect a very small mean square error (*less than 0.001*). 

Next, let's see how well the model is able to reconstruct the sequence:

~~~~python
pcc = Pcc(25,-1,1)
training = data[:round(len(data)/4)]
test = data[round(len(data)/4):]
for epoch in range(50):
    for trace,v in pcc.trace(training): 
        pcc.train(trace,v)
result = [v for v in pcc.gen(training,length=len(test))]
mse = np.square(np.array(result)-np.array(test)).mean()
mse50 = np.square(np.array(result[:50])-np.array(test[:50])).mean()
print('MSE full sequence: {0:.6f}'.format(mse))
print('MSE first 50: {0:.6f}'.format(mse50))
~~~~

This will approximately reproduce the sinus wave. Since the reconstruction will go out of phase with the test data, mse is expected to be close to 0.5. However, we also print the mse for the first 50 samples, this error is expected to be much smaller.

An important property of PCC is that it, although not reproducing an exact match of the original data, capture its basic properties. These are best illustrated by plotting the data. 

If you have matplotlib installed, you may try to visualize the reproduced data: 

~~~~python
fig,ax = plt.subplots(1,1)
ax.plot(result)
ax.plot(test)
~~~~

If we want a very precise long term generation, a more powerful model is needed. This can be chaieved by increasing the number of channels:

~~~~python
pcc = Pcc(200,-1,1) # Creating a model with 200 channels
training = data[:round(len(data)/4)]
test = data[round(len(data)/4):]
for epoch in range(200):
    for trace,v in pcc.trace(training): 
        pcc.train(trace,v)
result = [v for v in pcc.gen(training,length=len(test))]
mse = np.square(np.array(result)-np.array(test)).mean()
print('MSE full sequence: {0:.6f}'.format(mse))
~~~~

This should reproduce an mse close to 0.0. 

References
----------

Please refer to http://cognitionreversed.com for more examples and references. 

E. A. Billing. *Cognition Rehearsed: Recognition and Reproduction of Demonstrated Behaviour*. PhD thesis, Ume ̊a University, Sweden, 2012.

E. A. Billing, T. Hellström, and L.-E. Janlert. Predictive learning from demonstration. In J. Filipe, A. Fred, and B. Sharp, editors, *Proc. Second International Conference on Agents and Artificial Intelligence ICAART 2010*, volume CCIS 129, pages 186–200, Berlin Heidelberg, 2011. Springer-Verlag.

E. A. Billing, T. Hellstroö̈m, and L.-E. Janlert. Robot learning from demonstration using predictive sequence learning. In A. Dutta, editor, *Robotic Systems – Applications, Control and Programming*, pages 235–250. Intech, 2012.

E. A. Billing, H. Svensson, R. Lowe, and T. Ziemke. Finding your way from the bed to the kitchen: reenacting and recombining sensorimotor episodes learned from human demonstration. *Frontiers in Robotics and AI*, 3(9), 2016.
 

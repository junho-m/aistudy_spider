# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:23:42 2020

@author: 82103
"""


from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
    
    
X = np.linspace(0, 10, 10)
Y = X + np.random.randn(*X.shape)

for x, y in zip(X,Y):
    print((round(x,1), round(y,1)))
    
model = Sequential()
model.add(Dense(input_dim=1, units=1, activation="linear", use_bias=False))

sgd = optimizers.SGD(lr=0.05)
model.compile(optimizer='sgd', loss='mse')

weights = model.layers[0].get_weights()
w = weights[0][0][0]

print('initial w is : ' + str(w))

model.fit(X, Y, batch_size=10,epochs=10, verbose=1)

weights = model.layers[0].get_weights()
w = weights[0][0][0]

print('trained w is : ' + str(w))

plt.plot(X, Y, label='data')
plt.plot(X, w*X, label='prediction')
plt.legend()
plt.show()

    
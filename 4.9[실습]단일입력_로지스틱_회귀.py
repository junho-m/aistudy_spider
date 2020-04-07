# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:36:19 2020

@author: 82103
"""


from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential()
model.add(Dense(input_dim=1, units = 1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])

X = np.array([-2, -1.5, -1, 1.25, 1.62, 2]) 
Y = np.array([0, 0, 0, 1, 1, 1])

model.fit(X, Y, epochs=300, verbose=0)

model.predict([-2, -1.5, -1, 1.25, 1.62, 2])

model.predict([-1000, 1000])

model.summary()

model.layers[0].weights

model.layers[0].get_weights()


# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:24:31 2020

@author: 82103
"""


import numpy as np
from IPython.display import Image

inputs = np.array([
    [ [1,0] ]
])

Image(url= "https://raw.githubusercontent.com/captainchargers/deeplearning/master/img/lstm_cell2.png", width=500, height=250)

import tensorflow as tf
tf.reset_default_graph()
tf.set_random_seed(777)

tf_inputs = tf.constant(inputs, dtype=tf.float32)
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=1)
outputs, state = tf.nn.dynamic_rnn(
    cell=lstm_cell, dtype=tf.float32, inputs=tf_inputs)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _output, _state = sess.run([outputs, state])
    print("output values")
    print(_output)
    print("\nmemory cell value ")
    print(_state.c)
    print("\nhidden state value ")
    print(_state.h)
    
    
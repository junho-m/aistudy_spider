# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:20:14 2020

@author: 82103
"""


import tensorflow as tf

T = 1.0
F = 0.0
bias = 1.0


def get_AND_data():
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    Y = [
        [F],
        [F],
        [F],
        [T]
    ]
    
    return X, Y

def get_OR_data():
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    Y = [
        [F],
        [T],
        [T],
        [T]
    ]
    
    return X, Y

def get_XOR_data():
    X = [
    [F, F, bias],
    [F, T, bias],
    [T, F, bias],
    [T, T, bias]
    ]
    
    Y = [
        [F],
        [T],
        [T],
        [F]
    ]
    
    return X, Y

#X, Y = get_AND_data()
#X, Y = get_OR_data()
X, Y = get_XOR_data()

W = tf.Variable(tf.random_normal([3, 1]))

def step(x):
    return tf.to_float(tf.greater(x, 0))

f = tf.matmul(X, W)
output = step(f)
error = tf.subtract(Y, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(X, error, transpose_a=True)
train = tf.assign(W, tf.add(W, delta))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    err = 1
    epoch, max_epochs = 0, 20
    while err > 0.0 and epoch < max_epochs:
        epoch += 1
        err = sess.run(mse)
        sess.run(train)
        print('epoch:', epoch, 'mse:', err)
        
    print("\nTesting Result:")
    print(sess.run([output]))
    
    

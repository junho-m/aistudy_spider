# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:28:13 2020

@author: 82103
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from IPython.display import Image

Image(url= "https://raw.githubusercontent.com/captainchargers/deeplearning/master/img/autoencoder1.png", width=500, height=250)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# we will use train data for auto encoder training
x_train = x_train.reshape(60000, 784)

# select only 300 test data for visualization
x_test = x_test[:300]
y_test = y_test[:300]
x_test = x_test.reshape(300, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize data
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale


Image(url= "https://raw.githubusercontent.com/captainchargers/deeplearning/master/img/autoencoder2.png", width=500, height=250)

# input
_input = tf.placeholder(tf.float32, [None, 28*28])    # value in the range of (0, 1)
# encoder
encoder = tf.layers.dense(_input, 128, tf.nn.tanh)
# bottleneck
bottleneck = tf.layers.dense(encoder, 3)
# decoder
decoder = tf.layers.dense(bottleneck, 128, tf.nn.tanh)
#output
_output = tf.layers.dense(decoder, 28*28, tf.nn.sigmoid)
# loss function
loss = tf.losses.mean_squared_error(labels=_input, predictions=_output)
# optimizor
train = tf.train.AdamOptimizer(0.002).minimize(loss)

tf.set_random_seed(777)

# initialize
init = tf.global_variables_initializer()

# train hyperparameters
epoch_cnt = 50
batch_size = 5000
iteration = len(x_train) // batch_size

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    print("train start...")
    for epoch in range(epoch_cnt):
        avg_loss = 0.0
        start = 0; end = batch_size
        for i in range(iteration):
            _, loss_ = sess.run([train, loss], 
                               feed_dict={_input: x_train[start: end]})
            start += batch_size; end += batch_size
            # Compute average loss
            avg_loss += loss_ / iteration
        print("epoch : "+str(epoch)+ " , train loss : "+str(avg_loss))
    # take compressed vector
    _bottleneck = sess.run(bottleneck, {_input: x_test})
    
    
# visualize in 3D plot
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8

fig = plt.figure(1)
ax = Axes3D(fig)

xs = _bottleneck[:, 0]
ys = _bottleneck[:, 1]
zs = _bottleneck[:, 2]

color=['red','green','blue','lime','white','pink','aqua','violet','gold','coral']

for x, y, z, label in zip(xs, ys, zs, y_test):
    c = color[int(label)]
    ax.text(x, y, z, label, backgroundcolor=c)
    
ax.set_xlim(xs.min(), xs.max())
ax.set_ylim(ys.min(), ys.max())
ax.set_zlim(zs.min(), zs.max())

plt.show()    
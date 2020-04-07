# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:16:14 2020

@author: 82103
"""


import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[4,2])
Y = tf.placeholder(tf.float32, shape=[4,1])


# 두개의 입력값을 받는 두개의 뉴론을 만듭니다.  
W1 = tf.Variable(tf.random_uniform([2,2]))
# 각각의 뉴론은 한개의 편향값을 갖습니다.
B1 = tf.Variable(tf.zeros([2]))
# 출력값으로 Z를 리턴하도록 합니다. sigmoid(W1 * X + B1)
Z = tf.sigmoid(tf.matmul(X, W1) + B1)

# Z를 입력값으로 받는 하나의 뉴론을 두번째 히든레이어에 만듭니다.
W2 = tf.Variable(tf.random_uniform([2,1]))
# 뉴론은 한개의 편향값을 갖습니다.
B2 = tf.Variable(tf.zeros([1]))
# 출력값으로 Y_hat을 리턴합니다. sigmoid(W2 * Z + B2)
Y_hat = tf.sigmoid(tf.matmul(Z, W2) + B2)

# cross entropy
loss = tf.reduce_mean(-1*((Y*tf.log(Y_hat))+((1-Y)*tf.log(1.0-Y_hat))))

# 경사하강법
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# 학습데이터를 만듭니다.
train_X = [[0,0],[0,1],[1,0],[1,1]]
train_Y = [[0],[1],[1],[0]]

# 텐서플로우 매개변수를 초기화를 선언합니다.
init = tf.global_variables_initializer()
# 학습을 시작합니다.
with tf.Session() as sess:
    # 매개변수를 초기화합니다.
    sess.run(init)
    print("train data: "+str(train_X))
    # 2만번의 반복학습을 진행합니다.
    for i in range(20000):
        sess.run(train_step, feed_dict={X: train_X, Y: train_Y})
        if i % 5000 == 0:
            print('Epoch : ', i)
            print('Output : ', sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y}))
    
    print('Final Output : ', sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y}))
    
    
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:55:06 2018

@author: Joyce
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#assumption regression model y=wx+b
x_data = np.random.rand(50).astype(np.float32)
y_data = .3*x_data-.7

#fittiing model
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
target = w*x_data + b


loss = tf.reduce_mean(tf.square(target-y_data))
optimizer = tf.train.GradientDescentOptimizer(.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(w),sess.run(b))
        plt.plot(x_data, y_data, 'ro', label='Original data')
        plt.plot(x_data, sess.run(w) * x_data + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()




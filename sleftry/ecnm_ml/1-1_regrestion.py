# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:21:36 2018

@author: Joyce
"""
from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
import pandas as pd


data = pd.read_csv("D:\TF TEST LEARNING\ls\ecnm_ml\ml_data_all.csv", sep=",")

train_x = data[['tw_VALUE']]
train_y = data[['tw_Adj_Close']]
x = train_x.astype('float32').as_matrix()
y = train_y.astype('float32').as_matrix()
y_=y/1000
"""
xbar = tf.reduce_mean(x)
ybar = tf.reduce_mean(y_)
print('x=',xbar,'y=',ybar)
sumxy = tf.reduce_sum((x-xbar)*(y-ybar))
sumxx = tf.reduce_sum(tf.square(x-xbar))
bhat = sumxy/sumxx
b0 = ybar-bhat*xbar
print('b=',bhat,'b0=',b0)
"""
#w= tf.Variable(tf.random_uniform([1], -100.0, 100.0))
#b = tf.Variable(tf.zeros([1]))
w = tf.Variable(rng.normal(), name="weight")
b = tf.Variable(rng.normal(), name="bias")

target = tf.add(tf.multiply(x, w), b)


cost = tf.reduce_mean(tf.square(target-y_))/2
optimizer = tf.train.GradientDescentOptimizer(.0001).minimize(cost)

init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(101):
    sess.run(optimizer)
    if step % 10 == 0:
        print(step, sess.run(w), sess.run(b),sess.run(cost))
        plt.plot(x, y_, 'ro', label='Original data')
        plt.plot(x, sess.run(w) * x + sess.run(b),'bo', label='Fitted line')
        plt.legend()
        plt.show()


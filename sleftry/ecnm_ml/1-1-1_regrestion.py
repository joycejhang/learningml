# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:46:11 2018

@author: Joyce
"""

from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
import pandas as pd

data = pd.read_csv('D:\TF TEST LEARNING\ls\ecnm_ml\ml_data_all.csv', sep=",")

train_X = data[['tw_VALUE']]
train_Y = data[['tw_Adj_Close']]
n_samples = train_X.shape[0]
X = train_X.astype('float32').as_matrix()
Y = train_Y.astype('float32').as_matrix()


W = tf.Variable(rng.normal(), name="weight")
b = tf.Variable(rng.normal(), name="bias")

target = tf.add(tf.multiply(X, W), b)


cost = tf.reduce_sum(tf.pow(target-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(.5).minimize(cost)


init = tf.global_variables_initializer()



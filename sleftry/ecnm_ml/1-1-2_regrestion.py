# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:53:14 2018

@author: Joyce
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('D:\TF TEST LEARNING\ls\ecnm_ml\ml_data_all.csv', sep=",")

train_x = data[['tw_VALUE']]
train_y = data[['tw_Adj_Close']]
features = {key:np.array(value) for key,value in dict(train_x).items()} 
targets = {key:np.array(value) for key,value in dict(train_y).items()}
n_samples = train_x.shape[0]

w = tf.Variable(np.random.normal(), name="weight")
b = tf.Variable(np.random.normal(), name="bias")

y_ = tf.add(tf.multiply(features, w), b)


cost = tf.reduce_sum(tf.pow(y_-targets, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(.5).minimize(cost)



 
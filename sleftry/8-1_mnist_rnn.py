# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:31:33 2018

@author: Joyce
"""


import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
#from libs.utils import weight_variable, bias_variable
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

print("Package loaded")

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1), name)
def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape = shape), name)

tf.reset_default_graph()


n_input = 28 # MNIST data input (image shape: 28*28)
n_steps = 28 # steps
n_hidden = 128 # number of neurons in fully connected layer 
n_classes = 10 # (0-9 digits)

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    "w_fc" : weight_variable([n_hidden, n_classes], "w_fc")
}
biases = {
    "b_fc" : bias_variable([n_classes], "b_fc") 
}

x_transpose = tf.transpose(x, [1, 0, 2])
#print("x_transpose shape: %s" % x_transpose.get_shape())

x_reshape = tf.reshape(x_transpose, [-1, n_input])
#print("x_reshape shape: %s" % x_reshape.get_shape())

x_split = tf.split(x_reshape, n_steps,0)
print("type of x_split: %s" % type(x_split))
print("length of x_split: %d" % len(x_split))
print("shape of x_split[0]: %s" % x_split[0].get_shape())


basic_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
h, states = tf.nn.static_rnn(basic_rnn_cell, x_split, dtype=tf.float32)
print("type of outputs: %s" % type(h))
print("length of outputs: %d" % len(h))
print("shape of h[0]: %s" % h[0].get_shape())
print("type of states: %s" % type(states))

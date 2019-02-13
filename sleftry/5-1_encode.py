# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:56:14 2018

@author: Joyce
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1), name)
def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape = shape), name)

def plot_scatter(x, labels, title, txt = False):
    plt.title(title)
    ax = plt.subplot()
    ax.scatter(x[:,0], x[:,1], c = labels)
    txts = []
    if txt:
        for i in range(10):
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    plt.show()

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
e_W_1 = weight_variable([784, 300], "e_W_1")
e_b_1 = bias_variable([300], "e_b_1")
e_layer1 = tf.nn.relu(tf.matmul(x, e_W_1) + e_b_1)

e_W_2 = weight_variable([300, 100], "e_W_2")
e_b_2 = bias_variable([100], "e_b_2")
e_layer2 = tf.nn.relu(tf.matmul(e_layer1, e_W_2) + e_b_2)

e_W_3 = weight_variable([100, 20], "e_W_3")
e_b_3 = bias_variable([20], "e_b_3")
code_layer = tf.nn.relu(tf.matmul(e_layer2, e_W_3) + e_b_3)

d_W_1 = weight_variable([20, 100], "d_W_1")
d_b_1 = bias_variable([100], "d_b_1")
d_layer1 = tf.nn.relu(tf.matmul(code_layer, d_W_1) + d_b_1)

d_W_2 = weight_variable([100, 300], "d_W_2")
d_b_2 = bias_variable([300], "d_b_2")
d_layer2 = tf.nn.relu(tf.matmul(d_layer1, d_W_2) + d_b_2)

d_W_3 = weight_variable([300, 784], "d_W_3")
d_b_3 = bias_variable([784], "d_b_3")
output_layer = tf.nn.relu(tf.matmul(d_layer2, d_W_3) + d_b_3)

#tf.pow(x,y) x^y del^2
loss = tf.reduce_mean(tf.pow(output_layer - x, 2))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

#correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(e_layer1,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init_op = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init_op)
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        #train_accuracy = accuracy.eval(feed_dict={x: batch[0], e_layer1: batch[1]})
        #test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, e_layer1: mnist.test.labels})
        #print("step %d, training accuracy %g, test accuracy %g" % (i, train_accuracy, test_accuracy))
        print("step %d, loss %g"%(i, loss.eval(feed_dict={x:batch[0]})))
    optimizer.run(feed_dict={x: batch[0]})
    
print("final loss %g" % loss.eval(feed_dict={x: mnist.test.images}))
"""
trainimg = mnist.train.images
trainlabel = mnist.train.labels
output_nd = output_layer.eval(feed_dict = {x: mnist.train.images})


for i in [0, 1, 2, 3, 4]:
    curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix 
    ae_img = np.reshape(output_nd[i,:], (28, 28))
    curr_label = np.argmax(trainlabel[i, :] ) # Label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.matshow(ae_img, cmap=plt.get_cmap('gray'))
    """
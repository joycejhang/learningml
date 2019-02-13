# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:54:34 2018

@author: Joyce
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:50:56 2018

@author: Joyce
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1), name)
def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape = shape), name)

def plot_n_reconstruct(origin_img, reconstruct_img, n = 10):

    plt.figure(figsize=(2 * 10, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(origin_img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruct_img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def build_sae():
    W_e_1 = weight_variable([784, 300], "w_e_1")
    b_e_1 = bias_variable([300], "b_e_1")
    h_e_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_e_1), b_e_1))

    W_e_2 = weight_variable([300, 30], "w_e_2")
    b_e_2 = bias_variable([30], "b_e_2")
    h_e_2 = tf.nn.sigmoid(tf.add(tf.matmul(h_e_1, W_e_2), b_e_2))

    W_d_1 = weight_variable([30, 300], "w_d_1")
    b_d_1 = bias_variable([300], "b_d_1")
    h_d_1 = tf.nn.sigmoid(tf.add(tf.matmul(h_e_2, W_d_1), b_d_1))

    W_d_2 = weight_variable([300, 784], "w_d_2")
    b_d_2 = bias_variable([784], "b_d_2")
    h_d_2 = tf.nn.sigmoid(tf.add(tf.matmul(h_d_1, W_d_2), b_d_2))
    
    return [h_e_1, h_e_2], [W_e_1, W_e_2, W_d_1, W_d_2], h_d_2

def kl_div(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho,rho_hat), logfunc(invrho, invrhohat))
    return logrho
    
def logfunc(x, x2):
    return tf.matmul( x, tf.log(tf.div(x,x2)))

tf.reset_default_graph()
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape = [None, 784])
h, w, x_reconstruct = build_sae()


alpha = 5e-6
beta = 7.5e-5
kl_div_loss = map(lambda x: tf.reduce_sum(kl_div(0.02, tf.reduce_mean(x,0))), h)

#kl_div_loss = tf.reduce_sum(kl_div(0.02, tf.reduce_mean(h[0],0)))
l2_loss = tf.reduce_sum(lambda x, y: x + y, map(lambda x: tf.nn.l2_loss(x), w))

loss = tf.reduce_mean(tf.pow(x_reconstruct - x, 2)) + alpha * l2_loss + beta * kl_div_loss
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
init_op = tf.global_variables_initializer()

sess.run(init_op)

for i in range(10000):
    batch = mnist.train.next_batch(60)
    if i%100 == 0:
        print("step %d, loss %g"%(i, loss.eval(feed_dict={x:batch[0]})))
    optimizer.run(feed_dict={x: batch[0]})
    
print("final loss %g" % loss.eval(feed_dict={x: mnist.test.images}))
"""
for h_i in h:
    print("average output activation value %g" % tf.reduce_mean(h_i).eval(feed_dict={x: mnist.test.images}))
test_size = 10
test_origin_img = mnist.test.images[0:test_size, :]
test_reconstruct_img = x_reconstruct.eval(feed_dict = {x: test_origin_img})
plot_n_reconstruct(test_origin_img, test_reconstruct_img)

"""
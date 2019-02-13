# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:54:45 2018

@author: Joyce
"""

import tensorflow as tf
#import numpy as np
#import matplotlib.pypolt as plo

#read data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)


def mnistr(learningrate,batch):
    #set model y=wx+b
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    tg = tf.placeholder(tf.float32, [None, 10])

    
    #train medel tg=wx+b
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tg, logits=y))
    train_step = tf.train.GradientDescentOptimizer(learningrate).minimize(cross_entropy)
        
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
   

    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(batch)
        sess.run(train_step,feed_dict={x:batch_xs,tg:batch_ys})
    
    #check accuracy
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(tg,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, tg: mnist.test.labels}))
    
mnistr(.5,100)
 
        


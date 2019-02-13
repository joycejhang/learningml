# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:48:17 2018

@author: Joyce
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:40:17 2018

@author: Joyce
"""
import tensorflow as tf
import os
import path
import matplotlib.patheffects as PathEffects
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#os.chdir("\TF TEST LEARNING\ls\MNIST_data" )
#os.makedirs("/tmp/model")
#os.makedirs("/tmp/model-subset")
current_dir = os.getcwd()
#print ("workspace : %s" % os.getcwd())
#sess = tf.InteractiveSession()

#y=wx+b
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name)

#2*2 max_pool convolution
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#use Relu activation fuction
def add_layer(X, W, B):
    h_conv = tf.nn.relu(conv2d(X, W) + B)
    return max_pool_2x2(h_conv)


#input image x resolution is 28x28, y dimension is 10
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

#first layer x fitting model y=wx+b 2 dimetion with 5*5 filter than  activation Relu than desdimesion 2*2 max_pool
layer1 = add_layer(x_image, weight_variable([5, 5, 1, 32], "w_conv1"), bias_variable([32], "b_conv1"))
#2 dimetion with 5*5 filter than activation Relu 
layer2 = tf.nn.relu(conv2d(layer1, weight_variable([5, 5, 32, 48], "w_conv2")) + bias_variable([48], "b_conv2"))
#2 dimetion with 5*5 filter than  activation Relu than desdimesion 2*2 max_pool
layer3 = add_layer(layer2, weight_variable([5, 5, 48, 64], "w_conv3"), bias_variable([64], "b_conv3"))

#Densely Connected 1024 neural node
W_fc1 = weight_variable([7 * 7 * 64, 1024], "w_fc1")
b_fc1 = bias_variable([1024], "b_fc1")
h_pool2_flat = tf.reshape(layer3, [-1, 7 * 7 * 64])

#fitting model 1
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
#avoide overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#desdimesion to 10
W_fc2 = weight_variable([1024, 10], "w_fc2")
b_fc2 = bias_variable([10], "b_fc2")
#fitting model 2
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#loss function use logits with softmax  classification to 10 dimetion
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
#redresion with adam gradient 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#accuracy y_conv & y_
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()
init_op = tf.global_variables_initializer()

test_size = 5000
test_data = mnist.test.images[0:test_size, :]
test_label = mnist.test.labels[0:test_size, :]
test_label_index = np.argmax(test_label, axis = 1) 


# 建立一些變數以及對應的名字．
tf.reset_default_graph()
v1 = tf.Variable([0.1, 0.1], name="v1")
v2 = tf.Variable([0.2, 0.2], name="v2")

# 建立所有 variables 的初始化 ops
init_op = tf.global_variables_initializer()

# 建立 saver 物件
saver = tf.train.Saver()

with tf.Session() as sess:
    #save_path = saver.save(sess, "/tmp/model-subset/model.ckpt") # 儲存模型到 /tmp/model.ckpt
    saver = tf.train.import_meta_graph('/tmp/model-subset/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/model-subset/'))
    sess.run(tf.global_variables_initializer())
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print("%s with value %s" % (v.name, sess.run(v)))

"""
with tf.Session() as sess:
    
    # 執行初始化
    sess.run(init_op)
    
    #重新指定 v2 的值
    ops = tf.assign(v2, [0.3, 0.3])
    sess.run(ops)
    
    print(sess.run(tf.global_variables()))
    # ... 中間略去許多模型定義以及訓練，例如可以是 MNIST 的定義以及訓練
    
    save_path = saver.save(sess, "/tmp/model-subset/model.ckpt") # 儲存模型到 /tmp/model.ckpt
    """

"""    
with tf.Session() as sess:
    saver.restore(sess, "/tmp/model/model.ckpt")
    print("Model restored.")
    print("all values %s" % sess.run(tf.global_variables()))
    print("v2 value : %s" % sess.run(v2))
    """
    
"""    
tf.reset_default_graph()
v1 = tf.Variable([0.1, 0.1], name="v1")
v2 = tf.Variable([0.4, 0.4], name="v2")
init_ops = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_ops)
    saver = tf.train.Saver({"my_v2": v2})
    save_path = saver.save(sess, "/tmp/model-subset/model.ckpt")
"""
    
   


    
    

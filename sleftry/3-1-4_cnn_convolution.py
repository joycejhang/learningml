# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:14:16 2018

@author: Joyce
"""

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add a fully collected layer
    Weights = weight_variable([in_size, out_size])
    biases = bias_variable([out_size])
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)



    # reshape the input to have batch size, width, height, channel size
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 5*5 patch size, input channel is 1, output channel is 32
    W_conv1 = weight_variable([5, 5, 1, 32])

    # bias, same size with the output channel
    b_conv1 = bias_variable([32])

    # the first convolutional layer with a max pooling layer
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #after pooling, we have a tensor with shape[-1, 14, 14, 32]

    # the weights and bias for the second layer, we will get 64 channels
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    # the second convolutional layer with a max pooling layer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # after pooling, we have a tensor with shape[-1, 7, 7, 64]

    # add a fully connected layer with 1024 neurons and use relu as the activation function
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
    h_fc1 = add_layer(h_pool2_flat, 7*7*64, 1024, tf.nn.relu)

    # we add dropout for the fully connected layer to avoid overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # finally, the output layer
    y_conv = add_layer(h_fc1_drop, 1024, 10, None)




    # loss function and so on
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start training, and we test our model every 100 steps
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(10000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print("step %d, training accuracy %g, test accuracy %g" % (i, train_accuracy, test_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # modify the dir path to your own dataset
    parser.add_argument('--data_dir', type=str, default='/tmp/mnist',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()

"""
作者：求是大肥羊
链接：https://www.jianshu.com/p/7c79693e8897
來源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
"""
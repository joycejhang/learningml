import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

print("Package loaded")

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1), name)
def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape = shape), name)

n_input = 28 # MNIST data input (image shape: 28*28)
n_steps = 28 # steps
n_hidden = 128 # number of neurons in fully connected layer 
n_classes = 10 # (0-9 digits)

tf.reset_default_graph()
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    "w_fc" : weight_variable([n_hidden, n_classes], "w_fc")
}
biases = {
    "b_fc" : bias_variable([n_classes], "b_fc") 
}
x_transpose = tf.transpose(x, [1, 0, 2])
x_reshape = tf.reshape(x_transpose, [-1, n_input])
x_split = tf.split(x_reshape, n_steps, 0)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
h, states = tf.nn.static_rnn(lstm_cell, x_split, dtype=tf.float32)

h_fc = tf.matmul(h[-1], weights['w_fc']) + biases['b_fc']
y_ = h_fc

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 100
init_op = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init_op)

variables_names =[v.name for v in tf.trainable_variables()]

for step in range(5000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = np.reshape(batch_x, (batch_size, n_steps, n_input))
    cost_train, accuracy_train, states_train, rnn_out = sess.run([cost, accuracy, states, h[-1]], feed_dict = {x: batch_x, y: batch_y})
    values = sess.run(variables_names)
    rnn_out_mean = np.mean(rnn_out)
    for k,v in zip(variables_names, values):
        if k == 'RNN/BasicLSTMCell/Linear/Matrix:0':
            w_rnn_mean = np.mean(v)

    if step < 1500:
        if step % 100 == 0:
            print("step %d, loss %.5f, accuracy %.3f, mean of lstm weight %.5f, mean of lstm out %.5f" % (step, cost_train, accuracy_train, w_rnn_mean, rnn_out_mean))
    else:
        if step%1000 == 0: 
            print("step %d, loss %.5f, accuracy %.3f, mean of lstm weight %.5f, mean of lstm out %.5f" % (step, cost_train, accuracy_train, w_rnn_mean, rnn_out_mean))
    optimizer.run(feed_dict={x: batch_x, y: batch_y})
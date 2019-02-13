# -*- coding: utf-8 -*-
"""
Created on 2017年5月21日

@author: weizhen
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# 加載matplotlib工具包，使用該工具包可以對預測的sin函數曲線進行繪圖
import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
mpl.use('Agg')
from matplotlib import pyplot as plt
learn = tf.contrib.learn
HIDDEN_SIZE = 30  # Lstm中隱藏節點的個數
NUM_LAYERS = 2  # LSTM的層數
TIMESTEPS = 10  # 循環神經網絡的截斷長度
TRAINING_STEPS = 10000  # 訓練輪數
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 訓練數據個數
TESTING_EXAMPLES = 1000  # 測試數據個數
SAMPLE_GAP = 0.01  # 采樣間隔
# 定義生成正弦數據的函數
def generate_data(seq):
    X = []
    Y = []
    # 序列的第i項和後面的TIMESTEPS-1項合在一起作為輸入;第i+TIMESTEPS項作為輸出
    # 即用sin函數前面的TIMESTPES個點的信息，預測第i+TIMESTEPS個點的函數值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE,state_is_tuple=True)
    return lstm_cell

# 定義lstm模型
def lstm_model(X, y):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    # 通過無激活函數的全連接層計算線性回歸，並將數據壓縮成一維數組結構
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    
    # 將predictions和labels調整統一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    
    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adagrad",
                                             learning_rate=0.1)
    return predictions, loss, train_op

# 進行訓練
# 封裝之前定義的lstm
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir="D:/TF TEST LEARNING/LSTM/model_2"))
# 生成數據
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))
# 擬合數據
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# 計算預測值
predicted = [[pred] for pred in regressor.predict(test_X)]

# 計算MSE
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is:%f" % rmse[0])

plot_predicted, = plt.plot(predicted, label='predicted',  linewidth=10)
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
plt.show()
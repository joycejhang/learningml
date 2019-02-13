# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:28:17 2019

@author: Joyce
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

rnn_unit=10         #隱層神經元的個數
lstm_layers=2       #隱層層數
input_size=5
output_size=1
lr=0.0006         #學習率
#——————————匯入資料———---------
f=open('tw.csv')
df=pd.read_csv(f)     #讀入股票資料
data=df.iloc[:,0:6].values  #取第3-10列
data[:1]


def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=3000):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #mean_y=np.mean(data_train[:,0],axis=0)
    #std_y=np.std(data_train[:,0],axis=0)
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #標準化
    train_x,train_y=[],[]   #訓練集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,1:6]
       y=normalized_train_data[i:i+time_step,0,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#獲取測試集
def get_test_data(time_step=20,test_begin=3001):
    data_test=data[test_begin:5185]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #標準化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size個sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,1:6]
       y=normalized_test_data[i*time_step:(i+1)*time_step,0]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,1:6]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,0]).tolist())
    return mean,std,test_x,test_y

#——————————定義神經網路變數————————————
#輸入層、輸出層權重、偏置、dropout引數

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')  

#—————————定義神經網路變數————————————
def lstmCell():
    #basicLstm單元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm

def lstm(X):    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要將tensor轉成2維進行計算，計算後的結果作為隱藏層的輸入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #將tensor轉成3維，作為lstm cell的輸入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#train model
def traindata(batch_size=60, time_step=20, train_begin=0, train_end=3000):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):#reuse=tf.AUTO_REUSE
        pred,_=lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    #saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.save(sess, r"D:/TF TEST LEARNING/LSTM/model_save2/")
        for i in range(50):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                 keep_prob: 0.5})
            if i % 10 ==0:
                print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'model_save2\\modle.ckpt'))
        print("The train has finished")
        

#————————————————預測模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):#reuse=tf.AUTO_REUSE
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #引數恢復
        module_file = tf.train.latest_checkpoint('D:\TF TEST LEARNING\LSTM\model_save2')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]],keep_prob:1})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[0]+mean[0]
        test_predict=np.array(test_predict)*std[0]+mean[0]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
        print("The deviation of this predict:",acc)
        #以折線圖表示結果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

traindata()
prediction()
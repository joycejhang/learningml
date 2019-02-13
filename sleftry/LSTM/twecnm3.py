# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:21:05 2019

@author: Joyce
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_num=6

output_size=1
input_size=all_num-output_size
#batch_size % step_size ==0
rnn_unit=10
hide_layers=2
#lr=0.005


original_data=open('tw.csv')
read_original_data=pd.read_csv(original_data)
#row:m*(choice data column:all_num=input_size+output_size)
#data=df.iloc[:,2:9].values  #取第3-10列
select_data=read_original_data.iloc[:,:all_num].values
#lable = high_list[1:] + end_high
#df['lable'] = lable
#select_data[:1]


def get_train_data(batch_size,time_step,train_begain,train_end):
    #train_n*all_num
    #[N=train_begain-train_end,input_size]
    intercept_data=select_data[train_begain:train_end]
    normalize_train_data=(intercept_data-np.mean(intercept_data,axis=0))/np.std(intercept_data,axis=0)
    #0,batch_size,batch_size*2,batch_size*3,...,normalize_train_data-timp_step
    #lenth=[(N-timp_step)/batch_size]+1
    batch_index=[]
    train_x,train_y=[],[]
    for i in range(len(normalize_train_data)-time_step):
        if i % batch_size==0:
            #list.append(obj)
            batch_index.append(i)
            #print("Updated List : ", batch_index)
        #20*input_size
        #[time_step,input_size]
        x=normalize_train_data[i:i+time_step,1:6]
        #add dimension
        #[time_step,output_size]
        y=normalize_train_data[i:i+time_step,0,np.newaxis]
        #array,matrix >> list
        #[(N-time_step),(time_step*input_size)]
        train_x.append(x.tolist())
        #[(N-time_step),(time_step*output_size)]
        train_y.append(y.tolist())
    batch_index.append(len(normalize_train_data)-time_step)
    return batch_index,train_x,train_y
        
        
    
    
#W_IN:input_size*rnn_unit,2-D
#W_OUT:rnn_unit*output_size,2-D
weights={
        'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
        'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
        }
#B_IN:rnn_unit*(?=n-time_step+1)
#B_OUT:output_size*(?=n-time_step+1)
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
        }
# tf.placeholder(dtype,shape=None,name=None)
#sess.run(***, feed_dict={input: **})
keep_prob=tf.placeholder(tf.float32,name='keep_prob')

def lstmcell():
    basic_lstm=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    #regularization,avoid overfitting value:0~1
    dropout=tf.nn.rnn_cell.DropoutWrapper(basic_lstm,output_keep_prob=keep_prob)
    return basic_lstm


def rnn_lstm(X):
    #(N-time_step+1)*time_step*input_size,3-D
    #[,time_step,input_size]
    #x可以是tensor，也可不是tensor，返回是一個tensor
    lengh_num=tf.shape(X)[0]
    #time_step
    time_step=tf.shape(X)[1]
    #input_size*rnn_unit,2-D
    weight_in=weights['in']
    #B_IN:rnn_unit*(?=n-time_step+1)
    #[rnn_unit,]
    bias_in=biases['in']
    #[(N-time_step+1)*time_step]*input_size,2-D
    #[,input_size]
    input_features=tf.reshape(X,[-1,input_size])
    #input value wx+b
    #[,input_size]*[input_size,rnn_unit]+[rnn_unit,]=[,rnn_unit]
    input_rnn=tf.matmul(input_features,weight_in)+bias_in
    #[,time_step,rnn_unit]
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    cell=tf.nn.rnn_cell.MultiRNNCell([lstmcell() for i in range(hide_layers)])
    initialize_state=cell.zero_state(lengh_num,dtype=tf.float32)
    #outputs, last_states = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float64,sequence_length=X_lengths,inputs=X)
    #batch_size,sequence_length=maxmun lengh(0-padding),embedding_size
    #outputs,last_states
    output_rnn,last_states=tf.nn.dynamic_rnn(cell,input_rnn,initial_state=initialize_state,dtype=tf.float32)
    #[?,rnn_unit]
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    #W_OUT:rnn_unit*output_size,2-D
    #[rnn_unit,output_siz]
    weight_out=weights['out']
    #B_OUT:output_size*(?=n-time_step+1)
    #[output_size,]
    bias_out=biases['out']
    #[,rnn_unit]*[rnn_unit,output_siz]+[output_size,]=[,output_size]
    predict_label=tf.matmul(output,weight_out)+bias_out
    return predict_label,last_states,weight_in,bias_in

def train_lstm(batch_size,time_step,train_begain,train_end,learning_rate):
    #empty array matrix:(N-time_step+1)*time_step*input_size,3-D
    #[,time_step,input_size]
    features_x=tf.placeholder(tf.float32,shape=[None,time_step,input_size])
    #[,time_step,output_size]
    label_y=tf.placeholder(tf.float32,shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begain,train_end)
    with tf.variable_scope("layer_lstm"):
        predict_label, _ ,weight_in,bias_in=rnn_lstm(features_x)
    loss=tf.reduce_mean(tf.square(tf.reshape(predict_label,[-1])-tf.reshape(label_y,[-1])))    
    train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    #save model number
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=0)
    loss_array=[]
    iteration_array=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #iteration
        for i in range(5):
            for step in range(len(batch_index)-1):
                # keep_prob:0~1(preserve all)                 
                _,loss_=sess.run([train_op,loss],feed_dict={features_x:train_x[batch_index[step]:batch_index[step+1]],
                                                            label_y:train_y[batch_index[step]:batch_index[step+1]],
                                                            keep_prob:0.5})   
            #print("Number of iterations:",i,"loss",loss_)
            loss_array.append(loss_)
            iteration_array.append(i)
        plt.plot(iteration_array,loss_array,color='#054E9F')
        plt.title("model train loss cuver")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()
        print("model save:", saver.save(sess,'model\\model.ckpt'))
    #print(r[0],r[1],r[2],r[3],r[4],r[5])
    #print(c[0],c[1],c[2],c[3],c[4],c[5])
        #rls.append(r.tolist())
        #cls.append(c.tolist())
   


        #print("model_save:", saver.save(sess, 'model_save2\\modle.ckpt'))

    
        
train_lstm(60,20,0,3000,0.005)
        
    
    
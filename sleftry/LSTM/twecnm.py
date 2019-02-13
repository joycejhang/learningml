# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:33:19 2019

@author: Joyce
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as plt

#read original data
od=open('tw.csv')
rod=pd.read_csv(od)
#choice data column 1~6(next_day_hightest,tw_Open,tw_High,tw_Low,tw_Close,tw_Adj_Close)
cdt=rod.iloc[:,:6].values
cdt[:1]

#train model set y=wx+b
def train_data():
    #separate train & test data
    data_train=cdt[0:3000]
    #normolize original datat to x,y
    #nrmlz_cdt=tf.div(tf.subtract(cdt,np.mean(cdt,axis=0)),np.std(cdt,axis=0))
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    train_x,train_y=[],[]
    #for i in range(len(normalized_train_data)-time_step):
    x=normalized_train_data[:,1:6]
    y=normalized_train_data[:,0,np.newaxis]
    #train data x have five columns:tw_Open,tw_High,tw_Low,tw_Close,tw_Adj_Close
    #train data y has one column:next_day_hightest
    train_x.append(x.tolist())
    train_y.append(y.tolist())
    return train_x,train_y
#test model set y=wx+b 
def test_data():
    data_test=cdt[3001:5185]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #標準化
    test_x,test_y=[],[]
    x=normalized_test_data[:,1:6]
    y=normalized_test_data[:,0]
    test_x.append(x.tolist())
    #test_y.append(y.tolist())
    test_y.extend(y.tolist())
    return mean,std,test_x,test_y
#use LSTM alogrithm learning economy relationshop
def lstm():
    
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:50:34 2018

@author: Joyce
"""


import tensorflow as tf
  
reader = tf.train.NewCheckpointReader("D:/tmp/model/model.ckpt")  
  
variables = reader.get_variable_to_shape_map()  
 

for v in variables: 
    w = reader.get_tensor(v)  
    print(type(w))  
    #print(w.shape) 
    #print (w[0]) 
    print(w)
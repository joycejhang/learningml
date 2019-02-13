# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:45:24 2018

@author: Joyce
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#加上稀疏的限制條件 (sparse constraint)
#Sparsity Regularization
#先設定一個值，然後讓平均神經元輸出值 (average output activation vlue) 越接近它越好，如果偏離這個值，cost 函數就會變大，達到懲罰的效果
rho_hat = np.linspace(0 + 1e-2, 1 - 1e-2, 100)
rho = 0.2
#Kullback-Leibler divergence (relative entropy)
# rho_hat 為 0.2，而 rho 等於 0.2 的時候 kl_div = 0，rho 等於其他值時 kl_div 大於 0．
kl_div = rho * np.log(rho/rho_hat) + (1 - rho) * np.log((1 - rho) / (1 - rho_hat))
plt.plot(rho_hat, kl_div)
plt.xlabel("rho_hat")
plt.ylabel("kl_div")
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:37:24 2018

@author: Joyce
"""

from __future__ import print_function

import pandas as pd
pd.__version__

import numpy as np

#read econoemeic date
ecnm_dataframe = pd.read_csv("D:\TF TEST LEARNING\ls\ecnm_ml\ml_data_all.csv", sep=",")
#ecnm_dataframe.describe()

#choice before data
#ecnm_dataframe.head(100)

#tw add wight index distribution
#ecnm_dataframe.hist('tw_Adj_Close')

#set tw add wight index name tw_ind(tw_adj_close)
#set exchange rate tw/us name tw_us(tw_VALUE)
tw_Adj_Close = pd.Series()
tw_VALUE = pd.Series()
tw_ecnm = pd.DataFrame({'tw_indx':tw_Adj_Close,'tw_us':tw_VALUE})
"""
print(type(tw_ecnm['tw_indx']))
tw_ecnm['tw_indx']

print(type(tw_ecnm['tw_indx'][1]))
tw_ecnm['tw_indx'][1]

print(type(tw_ecnm[0:2]))
tw_ecnm[0:2]
"""

#use numpy
#np.log(tw_VALUE)

#use lambda
#y = tw_VALUE.apply(lambda val: val > 7000)
#tw_ecnm['tw_index_rate'] = (tw_ecnm['tw_indx'] > 7000) & tw_ecnm['tw_us'].apply(lambda name: name.startswith('San'))
#tw_ecnm

#other name with series
tw_ecnm['tw_indax_Open'] = pd.Series()
tw_ecnm['tw_index_delta'] = tw_ecnm['tw_indax_Open'] - tw_ecnm['tw_indx']
tw_ecnm

#reset order
tw_Adj_Close.index
tw_VALUE.index
tw_ecnm.index

#reset order with ?
tw_ecnm.reindex([2, 0, 1, 3, 4])

#random order with tw_ecnm
#DataFrame.reindex(labels=None, index=None, columns=None, axis=None, method=None, copy=True, level=None, fill_value=nan, limit=None, tolerance=None)[source]
#learning more http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reindex.html
tw_ecnm.reindex(np.random.permutation(tw_ecnm.index))
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 22:02:25 2018

@author: Joyce
"""

#add needed
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#read econoemeic date
ecnm_dataframe = pd.read_csv("D:\TF TEST LEARNING\ls\ecnm_ml\ml_data_4000.csv", sep=",")


def preprocess_features(ecnm_dataframe):
  """Prepares input features from economic data set.

  Args:
    ecnm_dataframe: A Pandas DataFrame expected to contain data
      from the economic data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  #
  selected_features = ecnm_dataframe[
    ["y","m","d","jp_Adj_Close","chnsse_Adj_Close","chnhs_Adj_Close","ind_Adj_Close","astsp_Adj_Close","ast_Adj_Close","usnsdq_Adj_Close","usvix_Adj_Close","ussp_Adj_Close","eurestx_Adj_Close","blx_Adj_Close","fnc_Adj_Close","grm_Adj_Close","cnd_Adj_Close","mxc_Adj_Close","chl_Adj_Close","tw_VALUE","chn_VALUE","chnhk_VALUE","bx_VALUE","cnd_VALUE","ind_VALUE","ko_VALUE","mlx_VALUE","mxc_VALUE","sd_VALUE","nafrc_VALUE","sgp_VALUE","sz_VALUE","aut_VALUE","eur_VALUE","nzn_VALUE","uk_VALUE","RDSCUNT_RATE","RATE_YEAR","RATE","BOND_RATE","PRP_M","PRP_M_P","M1A","M1B","M2","PUR_A","ASSETS","LIABILITIES"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  #processed_features["tw_delta"] = ((ecnm_dataframe["tw_High"]/1000.0) -(ecnm_dataframe["tw_Low"]/1000.0))
  return processed_features


def preprocess_targets(ecnm_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["tw_Adj_Close"] = (
    ecnm_dataframe["tw_Adj_Close"] / 1000.0)
  return output_targets


training_examples = preprocess_features(ecnm_dataframe.head(2000))
training_targets = preprocess_targets(ecnm_dataframe.head(2000))
validation_examples = preprocess_features(ecnm_dataframe.tail(2000))
validation_targets = preprocess_targets(ecnm_dataframe.tail(2000))


correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["tw_Adj_Close"]

y = correlation_dataframe.corr()
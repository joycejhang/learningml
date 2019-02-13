# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:14:23 2018

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
ecnm_dataframe = pd.read_csv("D:\TF TEST LEARNING\ls\ecnm_ml\ml_data_all.csv", sep=",")

#ecnm_dataframe = ecnm_dataframe.reindex(np.random.permutation(ecnm_dataframe.index))

def preprocess_features(california_housing_dataframe):
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
    ["y","m","d","tw_Open","tw_High","tw_Low","jp_Adj_Close","chnsse_Adj_Close","chnhs_Adj_Close","ko_Adj_Close","iny_Adj_Close","ind_Adj_Close","astsp_Adj_Close","ast_Adj_Close","usdow_Adj_Close","usnsdq_Adj_Close","usvix_Adj_Close","ussp_Adj_Close","eurestx_Adj_Close","blx_Adj_Close","fnc_Adj_Close","grm_Adj_Close","cnd_Adj_Close","mxc_Adj_Close","agt_Adj_Close","chl_Adj_Close","bx_Adj_Close","Isrl_Adj_Close","tw_VALUE","jp_VALUE","chn_VALUE","chnhk_VALUE","bx_VALUE","cnd_VALUE","ind_VALUE","ko_VALUE","mlx_VALUE","mxc_VALUE","sd_VALUE","nafrc_VALUE","sgp_VALUE","sz_VALUE","aut_VALUE","eur_VALUE","nzn_VALUE","uk_VALUE","RDSCUNT_RATE","RATE_YEAR","RATE","BOND_RATE","PRP_M","PRP_M_P","M1A","M1B","M2","PUR_A","ASSETS","LIABILITIES"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["tw_delta"] = ((ecnm_dataframe["tw_High"]/1000.0) -(ecnm_dataframe["tw_Low"]/1000.0))
  return processed_features

def preprocess_targets(california_housing_dataframe):
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
#training_examples.describe()

training_targets = preprocess_targets(ecnm_dataframe.head(2000))
#training_targets.describe()

vldlt = ecnm_dataframe.tail(3185)
vldft = vldlt.head(2000)
validation_examples = preprocess_features(vldft)
#validation_examples.describe()

validation_targets = preprocess_targets(vldft)
#validation_targets.describe()


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
      
def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["tw_Adj_Close"], 
      batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["tw_Adj_Close"], 
      num_epochs=1, 
      shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
      validation_examples, validation_targets["tw_Adj_Close"], 
      num_epochs=1, 
      shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor

linear_regressor = train_model(
    learning_rate=0.001,
    steps=1000,
    batch_size=1,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)


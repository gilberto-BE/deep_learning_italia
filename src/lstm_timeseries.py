# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:44:49 2020

@author: gilbe
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from kerastuner import HyperModel


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def multi_step_plot(history, true_future, prediction):
    STEP = 10
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo', label='True future')

    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future')

    plt.legend(loc='upper left')
    plt.show()


def data_proc():
    """Download data."""
    zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)
    return df


def univariate_data(dataset, start_index,
                    end_index, history_size,
                    target_size):
    """The parameter history_size is the size of the past window of
    information. The target_size is how far in the future does the model
    need to learn to predict. The target_size is the label
    that needs to be predicted."""
    data = []
    labels = []

    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size, )
        # to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt


def baseline(history):
    return np.mean(history)


def train_val_tf(x_train, y_train,
                 x_val, y_val,
                 batch_size=256,
                 buffer_size=10000):
    """
    Transform data to tensorflow format.
    """

    BATCH_SIZE = batch_size
    BUFFER_SIZE = buffer_size

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val = val.batch(BATCH_SIZE).repeat()

    return train, val


def multivariate_data(dataset, target, start_index,
                      end_index, history_size,
                      target_size, step,
                      single_step=False):
    """Single step predictions."""
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i: i + target_size])
    return np.array(data), np.array(labels)



def single_step_model(x_train_single, y_train_single,
                      x_val_single, y_val_single, epochs=10):
    """Train a multivariate LSTM, a step ahead."""

    EVALUATION_INTERVAL = 200

    train_data_single, val_data_single = train_val_tf(
        x_train_single, y_train_single,
        x_val_single, y_val_single)

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(
        tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    single_step_history = single_step_model.fit(
        train_data_single, epochs=epochs,
        steps_per_epoch=EVALUATION_INTERVAL,
        validation_data=val_data_single,
        validation_steps=50)

    return single_step_model, single_step_history


class TimeSeriesLSTM(HyperModel):

    """Subclass HyperModel in order to use hyperband.

    TODO: pass in hyperparameters as args
    """
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs

    def build(self, hp):
        """According to documentation you need to pass hp as argument.

        --Args:
            hp: Hyperparameter object.
        -- Input shape
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=hp.Int('units',
                                                    min_value=10,
                                                    max_value=200,
                                                    step=32),))
        model.add(tf.keras.layers.Dense(self.num_outputs))
        model.compile(tf.keras.optimizers.RMSprop(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
            loss='mae',
            metrics=['mae'])

        return model


def plot_preds(trained_model, val_data, take=3):
    """Get val_data from train_val_tf()"""
    for x, y in val_data.take(take):
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                        trained_model.predict(x)[0]],
                         12, 'Single Step Prediction')
        plt.show()


if __name__ == '__main__':
    data_proc()




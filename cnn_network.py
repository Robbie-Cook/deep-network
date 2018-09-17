"""
cnn_network.py

A neural network to use for pseudorehearsal

"""

import tensorflow as tf
from tensorflow import keras

import settings


"""
Make network
"""
def get_network():
    model = keras.Sequential() # Model is the model to return

    model.add(keras.layers.Dense(settings.numInputs, activation='relu', bias_initializer='RandomNormal'))
    model.add(keras.layers.Dropout(0.1)) # Dropout
    model.add(keras.layers.Dense(10, activation='sigmoid', bias_initializer='RandomNormal'))
    model.add(keras.layers.Dense(10, activation='relu', bias_initializer='RandomNormal'))
    model.add(keras.layers.Dense(1, activation='relu', bias_initializer='RandomNormal'))
    
    model.compile(optimizer=tf.train.AdagradOptimizer(learning_rate=0.1), loss=['mse'])



    return model


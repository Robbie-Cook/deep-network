"""
cnn_network.py

A neural network to use for pseudorehearsal

"""

import tensorflow as tf
from tensorflow import keras

import settings

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(settings.numInputs, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(settings.numOutputs, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=[])

"""
cnn_network.py

A neural network to use for pseudorehearsal

"""

import tensorflow as tf
from tensorflow import keras

import settings

model = keras.Sequential() # Model is the model to return

model.add(keras.layers.Dense(settings.numInputs, activation='sigmoid', bias_initializer='RandomNormal'))
for layer in range(settings.numHiddenLayers):
    # model.add(keras.layers.Dropout(0.2)) # Dropout
    model.add(keras.layers.Dense(settings.numHidden, activation='sigmoid', bias_initializer='RandomNormal'))
model.add(keras.layers.Dense(settings.numOutputs, activation='sigmoid', bias_initializer='RandomNormal'))

model.compile(
    tf.train.MomentumOptimizer(
        learning_rate=0.3,
        momentum=0.5,
        use_nesterov=False,
    ),
    loss = 'mse',
    metrics = []
)


"""
Network.py

A neural network to use for pseudorehearsal

"""

import tensorflow as tf
from tensorflow import keras

import settings

model = keras.Sequential() # Model is the model to return

activation = ('sigmoid' if not settings.reluLayers else 'relu')
model.add(keras.layers.Dense(settings.numInputs, activation=activation))
for layer in range(settings.numHiddenLayers):
    model.add(keras.layers.Dropout(settings.dropout)) # Dropout
    model.add(keras.layers.Dense(settings.numHidden, activation=activation, bias_initializer='RandomNormal'))
model.add(keras.layers.Dense(settings.numOutputs, activation=activation))

if settings.optimizer == tf.train.MomentumOptimizer:
    model.compile(
        tf.train.MomentumOptimizer(
            learning_rate=settings.learning_rate,
            momentum=settings.momentum,
            use_nesterov=False,
        ),
        loss = settings.loss,
        metrics = []
    )
elif settings.optimizer == tf.train.AdamOptimizer:
    model.compile(
        tf.train.AdamOptimizer(
            learning_rate=settings.learning_rate,
            beta1=settings.beta1,
            beta2=settings.beta2,
            epsilon=settings.epsilon
        ),
        loss = settings.loss,
        metrics = []
    )
else:
    raise Exception("Incorrect Optimiser")

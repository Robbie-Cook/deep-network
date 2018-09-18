"""
main_network.py

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

    model.add(keras.layers.Dense(settings.numInputs, activation='sigmoid', bias_initializer='RandomNormal'))
    for layer in range(settings.numHiddenLayers):
        model.add(keras.layers.Dropout(settings.dropout)) # Dropout
<<<<<<< HEAD
        model.add(keras.layers.Dense(settings.numHidden, activation='relu', bias_initializer='RandomNormal'))
=======
        model.add(keras.layers.Dense(settings.numHidden, activation= ('relu' if settings.reluLayers else 'sigmoid'), bias_initializer='RandomNormal'))
>>>>>>> 1109d00a88254cc268343fc78e54cfc5c928061a
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

    # model.compile(optimizer=tf.train.AdamOptimizer(), 
    #           loss='mse',
    #           metrics=[])



    return model


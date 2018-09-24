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

    model.add(keras.layers.Dense(settings.numHidden, activation='sigmoid', bias_initializer='RandomNormal'))
    for layer in range(settings.numHiddenLayers-1):
        model.add(keras.layers.Dropout(settings.dropout)) # Dropout
        model.add(keras.layers.Dense(settings.numHidden, activation= ('relu' if settings.reluLayers else 'sigmoid'), bias_initializer='RandomNormal'))
    model.add(keras.layers.Dense(settings.numOutputs, activation='sigmoid', bias_initializer='RandomNormal'))


    if not settings.adamOptimizer:
        model.compile(
            tf.train.MomentumOptimizer(
                learning_rate=0.3,
                momentum=0.5,
                use_nesterov=False,
            ),
            loss = 'mse',
            metrics = []
        )
    else:
        model.compile(
            keras.optimizers.Adam(),
            loss = 'mse',
            metrics = []
        )

    # model.compile(optimizer=tf.train.AdamOptimizer(), 
    #           loss='mse',
    #           metrics=[])



    return model


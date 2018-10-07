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
from tensorflow.keras.utils import plot_model
def get_network():
    model = keras.Sequential() # Model is the model to return

    model.add(keras.layers.Dense(settings.numHidden))
    for _ in range(settings.numHiddenLayers-1):
        model.add(keras.layers.Dropout(settings.dropout)) # Dropout
        model.add(keras.layers.Dense(settings.numHidden, activation= ('relu' if settings.reluLayers else 'sigmoid'), bias_initializer='RandomNormal'))
    model.add(keras.layers.Dense(settings.numOutputs))
    
    assert settings.modelType in ['classification', 'adam', 'normal'], "Incorrect model type"
    if settings.modelType == 'normal':
        model.compile(
            tf.train.MomentumOptimizer(
                learning_rate=0.1,
                momentum=0.3,
                use_nesterov=False,
            ),
            loss = 'mse',
            metrics = ['mae']
        )
    elif settings.modelType == 'adam':
        model.compile(
            keras.optimizers.Adam(),
            loss = 'mse',
            metrics = ['mae']
        )
    elif settings.modelType == 'classification':
        opt = keras.optimizers.Adam()
        model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['categorical_accuracy'])
    else:
        raise Exception("Error: Incorrect model type")

    # model.compile(optimizer=tf.train.AdamOptimizer(), 
    #           loss='mse',
    #           metrics=[])


    return model

    # model = keras.models.Sequential()
    # model.add(keras.layers.Dense(32))
    # model.add(keras.layers.Dense(1))
    # model.compile(loss='mse', optimizer='adam', metrics=['mae'])


    plot_model(model, to_file='test/model.png')

    return model